"""Preprocessing related prompts"""

GET_FEATURES_PROMPT = """
Là một mô hình AI, nhiệm vụ của bạn là xử lý dữ liệu đầu vào của vụ án pháp lý. Đầu vào sẽ bao gồm đoạn văn bản mô tả vụ án và tên của các đương sự. Vui lòng trích xuất các từ khóa từ phần mô tả và phân loại chúng vào 4 danh mục sau: Thông tin các bên, Hành vi dẫn đến tranh chấp, Đối tượng tranh chấp, Yếu tố lỗi và chứng cứ. Đầu ra phải là một đối tượng JSON và chỉ chứa JSON, không kèm theo bất kỳ văn bản, giải thích hoặc thông báo lỗi nào khác.

Giải thích từ khóa:
- Thông tin các bên: Trích xuất các thông tin liên quan đến nguyên đơn, bị đơn (ví dụ: cá nhân, tổ chức, chức vụ, vai trò trong giao dịch).
- Hành vi dẫn đến tranh chấp: Trích xuất loại hành vi thực tế (ví dụ: chậm thanh toán, vi phạm hợp đồng, không giao hàng).
- Đối tượng tranh chấp: Trích xuất đặc điểm tài sản, loại hình hợp đồng, quyền lợi bị vi phạm.
- Yếu tố lỗi và chứng cứ: Trích xuất các mô tả về ý chí chủ quan, sự vi phạm nghĩa vụ, và các chứng cứ liên quan.

Yêu cầu định dạng JSON:
- Sử dụng ngoặc kép cho cả khóa (key) và giá trị chuỗi (value).
- Mỗi khóa tương ứng với một danh mục, và giá trị là một mảng chuỗi chứa các từ khóa được trích xuất (nếu một danh mục không có từ khóa, hãy dùng mảng rỗng `[]`).
- Tên các khóa phải là: "parties_info", "dispute_acts", "subject_matter", "fault_and_evidence".

Ví dụ đầu ra (chỉ để tham khảo, đầu ra thực tế phải dựa trên nội dung đầu vào):
{
"parties_info": ["người cho vay", "bên vay", "công ty TNHH"],
"dispute_acts": ["không trả nợ đúng hạn", "chậm bàn giao nhà"],
"subject_matter": ["hợp đồng vay 500 triệu", "căn hộ chung cư"],
"fault_and_evidence": ["cố ý trốn tránh", "có giấy biên nhận nợ"]
}

Vui lòng đảm bảo CHỈ xuất ra đối tượng JSON.
Bây giờ hãy xử lý vụ án sau:
"""

CASE_SEG_PROMPT = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Nhiệm vụ của bạn là sắp xếp các mô tả thực tế khách quan về đương sự dựa trên phần mô tả vụ án và tên đương sự dưới đây.

### Định dạng đầu vào:
- Mô tả vụ án: {fact}
- Tên đương sự: {name}

### Lưu ý:
- **Dựa trên nội dung đầu vào**: Chỉ sắp xếp dựa trên phần mô tả vụ án được cung cấp, không thêm bất kỳ thông tin hoặc giả định bên ngoài nào.
- **Yêu cầu tính khách quan**: Phần mô tả phải hoàn toàn khách quan, tránh đưa vào kết quả phán quyết, đánh giá pháp lý hoặc phân tích chủ quan (chẳng hạn như suy diễn động cơ hoặc mang sắc thái cảm xúc).
- **Yêu cầu tính trọn vẹn**: Ngay cả khi một số hành vi không do đương sự trực tiếp thực hiện, nếu các hành vi này có liên quan đến đương sự (ví dụ: tạo thành nguyên nhân và kết quả, sự kiện bối cảnh, hoặc liên quan trực tiếp đến hành vi của đương sự), chúng cũng nên được đưa vào mô tả thực tế để đảm bảo bối cảnh đầy đủ.
- **Định dạng đầu ra**: Trực tiếp xuất ra phần mô tả thực tế khách quan đã được sắp xếp, nội dung phải ngắn gọn và chính xác, tránh thêm các phần giới thiệu, tóm tắt hoặc bình luận không liên quan.
- **Giới hạn tập trung**: Trọng tâm của mô tả nên nằm ở hành vi, vai trò và các sự kiện liên quan của đương sự, không đề cập đến các bên không liên quan khác hoặc các chi tiết phụ, trừ khi chúng có mối liên hệ rõ ràng với đương sự.

Vui lòng xử lý thông tin đầu vào theo các yêu cầu trên.
"""

PRE_JUDGE_PROMPT = """
Là một chuyên gia phân tích pháp lý, vui lòng phân tích vụ án sau đây một cách nghiêm ngặt dựa trên luật áp dụng, và xuất ra các vấn đề pháp lý có thể xảy ra theo các quy tắc sau:
1. Chỉ xuất ra các vấn đề có khả năng hợp lý (độ tin cậy > 30%)
2. Sắp xếp theo mức độ khả năng từ cao xuống thấp
3. Nếu có một vấn đề chính rõ ràng (độ tin cậy > 70%), ưu tiên xuất ra vấn đề đó. Nếu bạn rất chắc chắn rằng vấn đề đó là duy nhất, chỉ xuất ra vấn đề đó.
4. Nếu khả năng của các vấn đề khác < 10%, hãy loại trừ chúng.
5. Đầu ra phải ở định dạng mảng (list) Python: ['vấn đề 1', 'vấn đề 2', ...]

Đảm bảo đầu ra bắt đầu bằng "[" và chỉ chứa các vấn đề ứng viên phù hợp với chi tiết vụ án.
Nếu mô tả không phù hợp với bất kỳ vấn đề nào, xuất ra một mảng rỗng []. Không thêm bất kỳ văn bản giải thích hoặc chữ nào khác.

Vui lòng phân tích vụ án sau:
{case_text}
"""

__all__ = [
    "GET_FEATURES_PROMPT",
    "CASE_SEG_PROMPT",
    "PRE_JUDGE_PROMPT",
]
