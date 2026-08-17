import os
import random
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.utils.logger import logger

# Create a mock module for langchain_community.chat_models.vertexai
mock_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
# Just mock the ChatVertexAI class
mock_vertexai.ChatVertexAI = type("ChatVertexAI", (object,), {})

# Insert into sys.modules BEFORE importing ragas
sys.modules["langchain_community.chat_models.vertexai"] = mock_vertexai

from ragas import evaluate  # noqa: E402

try:
    from ragas.metrics.collections import Faithfulness, AnswerRelevancy  # noqa: E402
except ImportError:
    pass

load_dotenv()


def collect_data_for_ragas(rag, test_cases, method="vector"):
    rag.config.retrieve.to_dict = lambda: (
        {
            "method": method,
            "top_retrieve": True,
            "top_retrieve_top_k": 3,
            "direct_retrieve": True,
            "direct_retrieve_top_k": 3,
            "augment_retrieve": False,
        }
        if method == "graph"
        else {"method": "vector", "direct_retrieve_top_k": 3}
    )

    questions = []
    answers = []
    contexts_list = []

    for case in test_cases:
        case_input = {"fact": case["fact"], "name": ["Nguyên đơn", "Bị đơn"]}
        results = rag.analyze_case(case_input)

        # Merge outputs from all defendants/plaintiffs
        final_answer = ""
        final_contexts = []
        for r in results:
            final_answer += f"[{r['name']}]: {r.get('judge_result', '')}\n"

            # Extract contexts (texts of used laws and facts)
            for law in r.get("used_laws", []):
                final_contexts.append(law.get("description", "") or law.get("text", ""))
            for fact in r.get("used_facts", []):
                final_contexts.append(fact.get("fact", ""))

        if not final_contexts:
            # If no context, add a dummy to avoid ragas error
            final_contexts.append("Không có tài liệu tham khảo nào.")

        questions.append(case["fact"])
        answers.append(final_answer)
        contexts_list.append(final_contexts)

    return {"question": questions, "answer": answers, "contexts": contexts_list}


def main():
    config = LegalGraphRAGConfig()
    model_name = os.getenv("model_name", "gpt4o_mini")
    config.model.model_name = model_name

    api_key = os.getenv(f"{model_name}_api_key", os.getenv("api_key"))
    base_url = os.getenv(f"{model_name}_base_url", os.getenv("base_url"))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
    if base_url:
        os.environ["OPENAI_API_BASE"] = base_url

    config.model.api_key = api_key
    config.model.base_url = base_url
    config.model.device = "cpu"
    config.graph.graph_db_path = os.getenv("graph_db_path", "./data/processed/graph.pkl")

    logger.info("Initializing RAG for RAGAS evaluation...")
    rag = LegalGraphRAG(config)

    cases_db = rag.cases_db
    if not cases_db:
        logger.warning("cases_db in rag is None, manually loading...")
        try:
            cases_db = rag._load_cases_db()
            rag.cases_db = cases_db
        except Exception as e:
            logger.error(f"Failed to load cases_db: {e}")
            return
    valid_cases = [c for c in cases_db if c.get("law")]

    num_samples = 2
    test_cases = random.sample(valid_cases, num_samples)

    # 1. Run Vector RAG
    logger.info("Running Vector Retrieval...")
    vector_data = collect_data_for_ragas(rag, test_cases, "vector")

    # 2. Run Graph RAG
    logger.info("Running Graph Retrieval...")
    graph_data = collect_data_for_ragas(rag, test_cases, "graph")

    # Evaluate with Ragas
    logger.info("Starting RAGAS Evaluation...")

    # Configure LLM for Ragas to use the same OpenAI-compatible API
    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini" if model_name == "gpt4o_mini" else model_name,
        api_key=api_key,
        base_url=base_url,
    )
    evaluator_embeddings = OpenAIEmbeddings(
        api_key=api_key, base_url=base_url, model="text-embedding-3-small"
    )

    # Some APIs might fail with embeddings if it's not actually OpenAI, but we'll let Ragas try.
    # If the user is using DeepSeek or a local LLM, Answer Relevancy might fail because it requires embeddings.
    try:
        metrics = [Faithfulness(), AnswerRelevancy()]
    except Exception:
        # Fallback for older versions
        try:
            from ragas.metrics import faithfulness, answer_relevancy

            metrics = [faithfulness, answer_relevancy]
        except ImportError:
            pass

    try:
        logger.info("Evaluating Vector RAG...")
        vector_dataset = Dataset.from_dict(vector_data)
        vector_result = evaluate(
            vector_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        print("\n=== VECTOR RAGAS RESULT ===")
        print(vector_result)
        vector_result.to_pandas().to_csv("outputs/vector_ragas_detailed.csv", index=False)
        logger.info("Saved detailed Vector RAG results to outputs/vector_ragas_detailed.csv")
    except Exception as e:
        logger.error(f"Vector Ragas Error: {e}")

    try:
        logger.info("Evaluating Graph RAG...")
        graph_dataset = Dataset.from_dict(graph_data)
        graph_result = evaluate(
            graph_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        print("\n=== GRAPH RAGAS RESULT ===")
        print(graph_result)
        graph_result.to_pandas().to_csv("outputs/graph_ragas_detailed.csv", index=False)
        logger.info("Saved detailed Graph RAG results to outputs/graph_ragas_detailed.csv")
    except Exception as e:
        logger.error(f"Graph Ragas Error: {e}")


if __name__ == "__main__":
    main()
