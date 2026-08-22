import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()

from core.graph_construct.neo4j_manager import Neo4jManager  # noqa: E402
from core.graph_construct.llm_utils import get_embedding  # noqa: E402
from core.utils.logger import logger  # noqa: E402


def auto_link_unresolved_cases():
    logger.info("=" * 70)
    logger.info("BẮT ĐẦU TỰ ĐỘNG LIÊN KẾT 100% VỤ ÁN VÀ ĐIỀU LUẬT (AUTO-LINKING PIPELINE)")
    logger.info("=" * 70)

    cases_path = "data/clean/cases_clean.json"
    law_path = "data/clean/law_to_dispute_clean.json"

    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    with open(law_path, "r", encoding="utf-8") as f:
        laws = json.load(f)

    law_entries = {str(law_item.get("id", "")).strip(): law_item for law_item in laws}

    unlinked_cases = []
    for c in cases:
        c_laws = c.get("law", [])
        if isinstance(c_laws, str):
            c_laws = [c_laws]
        m = False
        for law_entry in c_laws:
            l_str = str(law_entry).strip()
            if l_str in law_entries or l_str.replace("zalo_", "") in law_entries:
                m = True
                break
        if not m:
            unlinked_cases.append(c)

    logger.info(
        f"Tổng số vụ án: {len(cases)} | Số vụ án chưa khớp điều luật: {len(unlinked_cases)}"
    )

    if not unlinked_cases:
        logger.info("Tất cả 100% vụ án đã được liên kết chính xác!")
        return

    neo4j = Neo4jManager()
    if not neo4j.driver:
        logger.error("Không thể kết nối tới Neo4j để tìm kiếm vector.")
        return

    resolved_count = 0
    with neo4j.driver.session() as session:
        for idx, case in enumerate(unlinked_cases, 1):
            fact_text = case.get("fact", "")
            dispute_list = case.get("dispute", [])
            query_text = f"{' '.join(dispute_list)} {fact_text[:300]}"

            emb = get_embedding(query_text)
            if not emb:
                continue

            # Query the best matching law in Neo4j vector index
            query = """
            CALL db.index.vector.queryNodes('law_embeddings', 3, $emb)
            YIELD node AS law, score
            RETURN law.id AS law_id, law.entry AS entry, law.description AS description, score
            ORDER BY score DESC
            LIMIT 1
            """
            result = session.run(query, emb=emb).single()
            if result and result["entry"]:
                best_entry = result["entry"]
                law_node_id = result["law_id"]
                score = result["score"]

                # Cập nhật mảng law của case
                old_law = case.get("law", [])
                case["law"] = [best_entry]
                case["law_resolved"] = [
                    {
                        "raw": old_law[0] if old_law else "",
                        "corpus_id": best_entry,
                        "score": score,
                        "confidence": "vector_auto_linked",
                    }
                ]
                case["law_resolve_status"] = "auto_resolved"

                # Nối cạnh RELATES_TO_LAW trực tiếp trên Neo4j
                edge_query = """
                MATCH (c:Cases), (l:Laws)
                WHERE (c.id = $case_id OR c.caseId = $case_id)
                  AND (l.id = $law_id OR l.entry = $law_id)
                MERGE (c)-[r:RELATES_TO_LAW]->(l)
                SET r.source = 'auto_link', r.score = $score
                """
                session.run(edge_query, case_id=case["id"], law_id=law_node_id, score=score)

                resolved_count += 1
                if idx % 20 == 0 or idx == len(unlinked_cases):
                    logger.info(
                        f"[{idx}/{len(unlinked_cases)}] Đã liên kết: Case {case['id']} -> {best_entry} (Score: {score:.4f})"
                    )

    # Lưu lại file cases_clean.json cập nhật
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    logger.info("=" * 70)
    logger.info(
        f"HOÀN THÀNH: Đã tự động liên kết thành công {resolved_count}/{len(unlinked_cases)} vụ án!"
    )
    logger.info(f"Đã lưu dữ liệu cập nhật vào {cases_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    auto_link_unresolved_cases()
