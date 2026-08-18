import sys
import os
from pprint import pprint
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.utils.util import analyze_case


def main():
    print("🚀 Initializing LegalGraphRAG...")
    # Load config from .env
    config = LegalGraphRAGConfig.from_env_file(".env")
    rag = LegalGraphRAG(config)

    # Ensure Neo4j and graphs are connected
    if not rag.cases_db:
        print("❌ Failed to load cases_db!")
        return

    case = rag.cases_db[0]
    if "name" not in case:
        case["name"] = "Người khiếu nại"

    print(f"✅ Loaded Case: {case.get('name', 'Unknown')}")
    print(f"📖 Fact preview: {case.get('fact', '')[:150]}...\n")

    # Base configuration
    retrieve_config = config.retrieve.to_dict()

    # 🌟 ENABLE ALL NEW PIPELINE FEATURES 🌟
    retrieve_config.update(
        {
            # K1: Cross-Encoder Reranker
            "use_reranker": True,
            "reranker_top_k": 8,
            # K4: Self-Consistency Judge with Gemini
            "use_self_consistent": True,
            "self_consistent_n": 3,
            "judge_chatbot": "gemini_flash_lite",
            # M1: Maximal Marginal Relevance (Diversity)
            "use_mmr": True,
            "mmr_lambda": 0.5,
            "mmr_top_k": 5,
            # M2: Score Thresholding
            "min_rrf_score": 0.01,
            # M4: Description Truncation
            "law_desc_cap": 500,
            # M5: Easy-first & Early Exit
            "max_applicable_laws": 3,
            # M7: Vietnamese Abbreviation Expansion
            "expand_abbreviations": True,
            # M8: RRF k tuning
            "rrf_k": 30,
            # Limits
            "max_judge_laws": 8,
        }
    )

    print("⚙️ Pipeline Configuration:")
    for k, v in retrieve_config.items():
        if k not in [
            "top_retrieve",
            "direct_retrieve",
            "augment_retrieve",
            "top_retrieve_top_k",
            "direct_retrieve_top_k",
        ]:
            print(f"  - {k}: {v}")

    print("\n⏳ Running analyze_case() with all new features enabled...")
    start_time = time.time()

    # MOCK THE RETRIEVER to bypass Neo4j network issues
    from core.retriever.graph_retriever import GraphRetriever

    def mock_retrieve(self, item, law_to_dispute, cases_db, config):
        dummy_laws = [
            {
                "id": "584",
                "entry": "Điều 584",
                "description": "Người nào có hành vi xâm phạm tính mạng, sức khỏe, danh dự, nhân phẩm, uy tín, tài sản, quyền, lợi ích hợp pháp khác của người khác mà gây thiệt hại thì phải bồi thường.",
                "judge_dep": ["Có hành vi trái pháp luật", "Có thiệt hại xảy ra"],
            },
            {
                "id": "585",
                "entry": "Điều 585",
                "description": "Thiệt hại thực tế phải được bồi thường toàn bộ và kịp thời.",
                "judge_dep": [],
            },
            {
                "id": "586",
                "entry": "Điều 586",
                "description": "Năng lực chịu trách nhiệm bồi thường thiệt hại của cá nhân.",
                "judge_dep": [
                    "Người từ đủ mười lăm tuổi đến chưa đủ mười tám tuổi gây thiệt hại thì phải bồi thường"
                ],
            },
            {
                "id": "589",
                "entry": "Điều 589",
                "description": "Thiệt hại do tài sản bị xâm phạm bao gồm tài sản bị mất, bị hủy hoại hoặc bị hư hỏng.",
                "judge_dep": ["Tài sản bị xâm phạm"],
            },
        ]
        dummy_facts = [
            {"id": "fact1", "description": "Tài sản của công ty bị mất", "law": ["584", "589"]}
        ]
        return {"fusion_method": "rrf (mock)"}, dummy_laws, dummy_facts

    GraphRetriever.retrieve = mock_retrieve

    # Run the pipeline
    results = analyze_case(
        chatbot=rag.model,
        case=case,
        law_to_dispute=rag.law_to_dispute,
        cases_db=rag.cases_db,
        retrieve_config=retrieve_config,
    )

    end_time = time.time()

    print(f"\n✅ Analysis complete in {end_time - start_time:.2f} seconds.")
    print("=" * 60)

    for i, res in enumerate(results):
        print(f"\n🧑‍⚖️ Defendant {i+1}: {res.get('name')}")
        print(f"Confidence: {res.get('confidence')}")

        print("\n🔍 Reasoning Trace (Telemetry):")
        pprint(res.get("reasoning_trace", {}))

        print(f"\n📚 Used Laws ({len(res.get('used_laws', []))}):")
        for law in res.get("used_laws", []):
            print(f"  - {law.get('id')}")

        print(f"\n⚖️ Final Judgment:\n{res.get('judge_result')}\n")


if __name__ == "__main__":
    main()
