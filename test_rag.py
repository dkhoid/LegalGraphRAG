"""
Test RAG Pipeline - Đa dạng loại tranh chấp
Chạy: python test_rag.py
"""

import sys
import os
import json
import time

sys.path.append(".")

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.graph_construct.graph_db import GraphDBManager

# ==========================================
# Bộ test cases đa dạng (10 loại tranh chấp)
# ==========================================
TEST_CASES = [
    {
        "id": "test_thừa_kế",
        "tag": "Thừa kế",
        "name": "Trần Văn B",
        "fact": "Bố tôi mất đột ngột không để lại di chúc. Gia đình có 4 anh chị em. Bố tôi để lại 1 căn nhà và 2 mảnh đất. Anh cả tự ý chiếm toàn bộ tài sản và không chịu chia cho các anh em khác. Tôi muốn hỏi pháp luật quy định thế nào về việc phân chia tài sản thừa kế khi không có di chúc?",
    },
    {
        "id": "test_hợp_đồng_mua_bán",
        "tag": "Hợp đồng mua bán",
        "name": "Lê Thị C",
        "fact": "Tôi đặt mua một lô hàng trị giá 200 triệu đồng từ công ty X. Hợp đồng ghi rõ giao hàng trong 30 ngày. Tuy nhiên sau 45 ngày họ vẫn chưa giao hàng và cũng không hoàn tiền cọc 50 triệu. Tôi có thể yêu cầu hủy hợp đồng và đòi bồi thường không?",
    },
    {
        "id": "test_lao_động_sa_thải",
        "tag": "Sa thải, chấm dứt HĐLĐ",
        "name": "Nguyễn Văn D",
        "fact": "Tôi làm việc tại công ty được 5 năm với hợp đồng lao động không xác định thời hạn. Hôm qua giám đốc gọi tôi lên và thông báo sa thải tôi ngay lập tức mà không có lý do chính đáng, cũng không báo trước 45 ngày. Tôi có quyền khiếu nại và được bồi thường gì không?",
    },
    {
        "id": "test_bồi_thường_thiệt_hại",
        "tag": "Bồi thường thiệt hại",
        "name": "Phạm Thị E",
        "fact": "Con trai tôi 15 tuổi đi xe đạp trên đường và bị một ô tô đâm gãy chân. Người lái xe bỏ chạy nhưng đã bị camera ghi lại biển số. Con tôi phải nằm viện 2 tháng, chi phí điều trị hết 80 triệu đồng. Tôi muốn khởi kiện đòi bồi thường thiệt hại.",
    },
    {
        "id": "test_thế_chấp_cầm_cố",
        "tag": "Thế chấp, cầm cố",
        "name": "Hoàng Văn F",
        "fact": "Tôi dùng sổ đỏ nhà đất thế chấp ngân hàng để vay 1 tỷ đồng. Do ảnh hưởng kinh tế tôi không trả được nợ đúng hạn. Ngân hàng thông báo sẽ phát mãi tài sản thế chấp. Tôi muốn hỏi ngân hàng có quyền tự phát mãi mà không thông qua tòa án không?",
    },
    {
        "id": "test_hợp_đồng_thuê",
        "tag": "Hợp đồng thuê",
        "name": "Đỗ Thị G",
        "fact": "Tôi cho thuê nhà với hợp đồng 2 năm, giá thuê 10 triệu/tháng. Bên thuê đã nợ tiền thuê 4 tháng liên tiếp và sử dụng nhà sai mục đích kinh doanh (hợp đồng ghi ở mục đích để ở). Tôi muốn chấm dứt hợp đồng thuê và yêu cầu bồi thường.",
    },
    {
        "id": "test_tiền_lương",
        "tag": "Tiền lương",
        "name": "Vũ Văn H",
        "fact": "Tôi làm việc tại nhà máy sản xuất được 3 năm. 2 tháng gần đây công ty không trả lương cho toàn bộ công nhân với lý do khó khăn tài chính. Tổng số tiền lương công ty nợ tôi là 24 triệu đồng. Tôi có quyền đơn phương chấm dứt hợp đồng và đòi lương không?",
    },
    {
        "id": "test_giao_dịch_dân_sự",
        "tag": "Giao dịch dân sự",
        "name": "Trịnh Văn I",
        "fact": "Tôi 17 tuổi tự ý ký hợp đồng mua xe máy trị giá 30 triệu đồng mà không có sự đồng ý của bố mẹ. Sau đó bố mẹ tôi yêu cầu trả lại xe và lấy lại tiền. Bên bán không đồng ý. Hợp đồng này có hiệu lực pháp lý hay không?",
    },
    {
        "id": "test_đại_diện",
        "tag": "Đại diện",
        "name": "Ngô Thị K",
        "fact": "Mẹ tôi bị tai biến mất năng lực hành vi dân sự. Tôi được tòa án chỉ định làm người giám hộ. Anh trai tôi tự ý bán mảnh đất đứng tên mẹ mà không có sự đồng ý của tôi với tư cách người giám hộ. Giao dịch mua bán đất này có hợp pháp không?",
    },
    {
        "id": "test_hợp_đồng_vay",
        "tag": "Hợp đồng vay tài sản",
        "name": "Bùi Văn L",
        "fact": "Tôi cho bạn vay 500 triệu đồng có giấy vay nợ viết tay, thỏa thuận trả trong 1 năm với lãi suất 1%/tháng. Đã quá hạn 6 tháng nhưng bạn tôi vẫn không trả. Tôi muốn kiện ra tòa để đòi lại tiền gốc và lãi.",
    },
]


def run_tests():
    GraphDBManager.load("data/clean/graph.pkl")

    config = LegalGraphRAGConfig.from_env_file(".env")
    rag = LegalGraphRAG(config=config)
    rag.cases_db = rag._load_cases_db()
    rag.law_to_dispute = rag._load_law_to_dispute()

    results = []

    for idx, tc in enumerate(TEST_CASES, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(TEST_CASES)}] 🏷️  Loại tranh chấp: {tc['tag']}")
        print(f"    Bên liên quan: {tc['name']}")
        print(f"    Tình tiết: {tc['fact'][:100]}...")
        print(f"{'='*70}")

        sample_case = {
            "id": tc["id"],
            "name": tc["name"],
            "fact": tc["fact"],
            "description": tc["fact"],
        }

        start_time = time.time()
        try:
            result = rag.analyze_case(sample_case)
            elapsed = time.time() - start_time

            for r in result:
                judge = r.get("judge_result", {})
                used_laws = r.get("used_laws", [])

                print(f"\n  ⚖️  Kết quả phân tích:")
                print(f"     Loại tranh chấp AI xác định: {judge.get('dispute_type', 'N/A')}")
                print(f"     Điều luật áp dụng: {judge.get('law_article', 'N/A')}")

                resolution = judge.get("resolution", {})
                if isinstance(resolution, dict):
                    print(f"     Trách nhiệm: {resolution.get('liability', 'N/A')[:150]}")
                    print(f"     Bồi thường:  {resolution.get('compensation', 'N/A')[:150]}")
                else:
                    print(f"     Hướng xử lý: {str(resolution)[:200]}")

                print(f"     Số luật trích xuất từ Graph: {len(used_laws)}")
                print(f"     ⏱️  Thời gian xử lý: {elapsed:.1f}s")

                results.append(
                    {
                        "id": tc["id"],
                        "tag": tc["tag"],
                        "dispute_type": judge.get("dispute_type", []),
                        "law_count": len(used_laws),
                        "time_seconds": round(elapsed, 1),
                        "status": "OK",
                    }
                )

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n  ❌ LỖI: {e}")
            results.append(
                {
                    "id": tc["id"],
                    "tag": tc["tag"],
                    "status": "ERROR",
                    "error": str(e),
                    "time_seconds": round(elapsed, 1),
                }
            )

    # Tổng kết
    print(f"\n\n{'='*70}")
    print(f"📊 TỔNG KẾT: {len(results)} test cases")
    print(f"{'='*70}")
    ok_count = sum(1 for r in results if r["status"] == "OK")
    err_count = sum(1 for r in results if r["status"] == "ERROR")
    total_time = sum(r["time_seconds"] for r in results)
    print(f"  ✅ Thành công: {ok_count}/{len(results)}")
    print(f"  ❌ Thất bại:   {err_count}/{len(results)}")
    print(f"  ⏱️  Tổng thời gian: {total_time:.1f}s (TB: {total_time/len(results):.1f}s/case)")

    with open("outputs/test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  📁 Kết quả chi tiết đã lưu tại: outputs/test_results.json")


if __name__ == "__main__":
    run_tests()
