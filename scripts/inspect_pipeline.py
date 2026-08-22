import sys
import json
import time
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.graph_construct.graph_db import GraphDBManager


def print_step(step_num, title):
    print(f"\n{'-'*60}")
    print(f"[STEP {step_num}] {title}")
    print(f"{'-'*60}")


def inspect():
    print("⏳ Khởi tạo LegalGraphRAG và GraphDB...")
    try:
        GraphDBManager.load("data/clean/graph.pkl")
    except Exception as e:
        print(f"Lỗi tải GraphDB: {e}")
        return

    config = LegalGraphRAGConfig.from_env_file(".env")

    # Cấu hình hoàn toàn tuân theo file .env (không ghi đè cứng)

    rag = LegalGraphRAG(config=config)
    rag.cases_db = rag._load_cases_db()
    rag.law_to_dispute = rag._load_law_to_dispute()

    print("\n✅ Hệ thống đã sẵn sàng.")
    print(
        "\n💡 GỢI Ý: Bạn có thể nhập một tình huống pháp lý (hoặc ấn Enter để dùng tình huống mẫu)."
    )

    user_fact = input("\n📝 Nhập tình tiết vụ việc (Fact): ").strip()
    if not user_fact:
        user_fact = "Tôi cho bạn vay 500 triệu đồng có giấy vay nợ viết tay, thỏa thuận trả trong 1 năm với lãi suất 1%/tháng. Đã quá hạn 6 tháng nhưng bạn tôi vẫn không trả. Tôi muốn kiện ra tòa để đòi lại tiền gốc và lãi."
        print(f"👉 Dùng tình huống mẫu: {user_fact}")

    user_name = input(
        "👤 Nhập tên bị đơn / người liên quan (ví dụ: Bạn tôi, Nguyễn Văn A): "
    ).strip()
    if not user_name:
        user_name = "Bạn tôi"
        print(f"👉 Dùng tên mẫu: {user_name}")

    sample_case = {
        "id": "inspect_test",
        "name": user_name,
        "fact": user_fact,
        "description": user_fact,
    }

    print("\n🚀 ĐANG CHẠY PHÂN TÍCH (Vui lòng đợi vài giây)...\n")

    try:
        start_time = time.time()
        results = rag.analyze_case(sample_case)
        elapsed = time.time() - start_time

        for r in results:
            print(f"\n{'='*70}")
            print(f" BÁO CÁO PHÂN TÍCH CHO ĐỐI TƯỢNG: {r.get('name', 'N/A')}")
            print(f"{'='*70}")

            # STEP 1: Feature Extraction
            print_step(1, "📝 TRÍCH XUẤT ĐẶC TRƯNG (Feature Extraction)")
            features = r.get("feature", {})
            if features:
                print(json.dumps(features, indent=2, ensure_ascii=False))
            else:
                print("⚠️ Không có đặc trưng nào được trích xuất.")

            # STEP 2: Retrieval
            print_step(2, "🔍 LẤY DỮ LIỆU TỪ GRAPH (Retrieval)")
            retrieved_laws = r.get("retrieved_laws", [])
            retrieved_facts = r.get("retrieved_facts", [])

            print(f"✅ Số lượng Tình tiết (Cases) lấy ra: {len(retrieved_facts)}")
            print(f"✅ Số lượng Điều luật (Laws) lấy ra: {len(retrieved_laws)}")

            if retrieved_laws:
                print("\n📑 TOP 3 Điều luật được hệ thống Graph/Reranker ưu tiên cao nhất:")
                for i, law in enumerate(retrieved_laws[:3], 1):
                    print(
                        f"  {i}. [Điều {law.get('entry', 'N/A')}] - {law.get('description', '')[:100]}..."
                    )

            if retrieved_facts:
                print("\n📂 TOP 3 Tình tiết (Cases) tương tự nhất:")
                for i, fact in enumerate(retrieved_facts[:3], 1):
                    print(f"  {i}. {fact.get('description', '')[:100]}...")

            # STEP 3: LLM Judge
            print_step(3, "⚖️ LLM JUDGE (Lọc và Đánh giá điều luật)")
            used_laws = r.get("used_laws", [])

            print(
                f"✅ Số điều luật LLM quyết định SỬ DỤNG (used_laws): {len(used_laws)} / {len(retrieved_laws)}"
            )

            if used_laws:
                print("\n📑 Các Điều luật được giữ lại để suy luận:")
                for i, law in enumerate(used_laws, 1):
                    print(f"  {i}. [Điều {law.get('entry', 'N/A')}]")

            if len(retrieved_laws) > len(used_laws):
                print(
                    f"\n❌ Đã lọc bỏ {len(retrieved_laws) - len(used_laws)} điều luật không thật sự liên quan."
                )

            # STEP 4: Final Output
            print_step(4, "🎯 KẾT QUẢ CUỐI CÙNG (Final Output)")
            judge = r.get("judge_result", {})

            print(f"🔹 Loại tranh chấp: {judge.get('dispute_type', 'N/A')}")
            print(f"🔹 Điều luật áp dụng: {judge.get('law_article', 'N/A')}")

            resolution = judge.get("resolution", {})
            if isinstance(resolution, dict):
                print(f"🔹 Trách nhiệm (Liability):\n   {resolution.get('liability', 'N/A')}")
                print(f"🔹 Bồi thường (Compensation):\n   {resolution.get('compensation', 'N/A')}")
            else:
                print(f"🔹 Hướng xử lý:\n   {str(resolution)}")

            print(f"\n⏱️ Tổng thời gian chạy: {elapsed:.1f} giây")

    except Exception as e:
        print(f"\n❌ LỖI TRONG QUÁ TRÌNH CHẠY: {e}")


if __name__ == "__main__":
    inspect()
