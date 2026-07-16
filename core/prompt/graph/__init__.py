"""Graph construction related prompts"""

SUMMARIZE_TEXTS_PROMPT = """
Bạn là một chuyên gia pháp lý giàu kinh nghiệm. Nhiệm vụ của bạn là trừu tượng hóa các danh mục vấn đề pháp lý tổng quát, cấp cao từ một cộng đồng gồm các vụ án. Vui lòng tuân thủ các quy tắc sau:

1. **Phân tích và Quy nạp**: Cẩn thận phân tích tất cả các hành vi đầu vào, xác định bản chất cốt lõi và kiểu mẫu chung của chúng.
2. **Khái quát hóa cấp cao**: Đầu ra phải là một mô tả danh mục duy nhất được tinh chỉnh cao, không phải là việc liệt kê hoặc lặp lại các hành vi đầu vào.
3. **Định dạng đầu ra**: Đầu ra phải tuân thủ nghiêm ngặt định dạng sau và chỉ nằm trên một dòng: "Vấn đề pháp lý: [Danh mục tổng quát]".
4. **Các điều cấm**: Không xuất ra bất kỳ chi tiết hành vi cụ thể, văn bản giải thích, danh sách, hay thông tin bổ sung nào.

    """

RERANK_CLUSTERS_PROMPT_TEMPLATE = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp, giỏi lập bản đồ các mô tả vụ án cụ thể sang các danh mục vấn đề pháp lý cấp cao.

Yêu cầu xử lý:
- Phân tích kỹ các hành vi chính, ý định chủ quan và các quan hệ pháp luật liên quan trong vụ án.
- So sánh với mô tả đặc điểm của từng danh mục và đánh giá mức độ phù hợp của mỗi danh mục.
- Sắp xếp tất cả các danh mục theo độ liên quan từ cao xuống thấp.
- Định dạng đầu ra: ví dụ: "rank: [3,1,2]", không được phép có văn bản bổ sung nào khác.

Bây giờ hãy xử lý thông tin sau:

Tóm tắt các danh mục có sẵn:
{cluster_summaries}

Mô tả vụ án cần phân tích:
{query_text}
    """

RERANK_PROMPT_TEMPLATE = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Tôi cần bạn xếp hạng lại các vụ án tương tự này theo độ liên quan của chúng với vụ án gốc, và đưa ra ba vụ án liên quan nhất.

Mô tả nhiệm vụ:
1. Phân tích mức độ liên quan giữa từng vụ án tương tự (codeX) và vụ án gốc
2. Sắp xếp lại các vụ án tương tự theo thứ tự từ cao xuống thấp
3. Đưa ra số thứ tự của tối đa ba vụ án liên quan nhất
4. Định dạng đầu ra phải là một danh sách số nguyên, chỉ chứa phần số của các mã vụ án.

Yêu cầu đầu ra:
- Chỉ xuất ra một danh sách số nguyên, định dạng như sau: [3, 1, 2]
- Các số trong danh sách tương ứng với mã số của các vụ án tương tự (số đằng sau từ code)
- Vụ án đứng đầu là vụ án liên quan nhất tới vụ án gốc

Thông tin vụ án tương tự:
{neighbor_summaries}

Nội dung vụ án gốc:
{query_text}

Vui lòng xuất danh sách mã số vụ án đã xếp hạng:
    """

__all__ = [
    "SUMMARIZE_TEXTS_PROMPT",
    "RERANK_CLUSTERS_PROMPT_TEMPLATE",
    "RERANK_PROMPT_TEMPLATE",
]
