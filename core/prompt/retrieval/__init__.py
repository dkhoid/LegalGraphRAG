"""Retrieval related prompts"""

RETRIEVE_LAW_PROMPT = """
Bạn là một chuyên gia phân tích pháp lý cần phân tích tối đa ba vấn đề pháp lý có thể xảy ra dựa trên các tình tiết vụ án. Vui lòng tuân thủ các yêu cầu sau:

**Yêu cầu phân tích:**
1. Phân tích toàn diện mọi khía cạnh hành vi trong tình tiết vụ án
2. Xem xét tất cả các vấn đề pháp lý mà hành vi có thể liên quan
3. Bao gồm cả các vấn đề chính và các vấn đề phụ/liên quan

**Yêu cầu đầu ra:**
- Chỉ xuất theo định dạng list Python: ["vấn đề 1", "vấn đề 2", ...]
- Sắp xếp theo khả năng xảy ra từ cao xuống thấp
- Đưa ra tối đa ba vấn đề
- Không giải thích, không đánh số, không thêm bất kỳ nội dung nào khác

**Thông tin vụ án:**
Đương sự: {name}
Tình tiết vụ án:
```
{fact}
```
Bây giờ vui lòng xuất ra:
"""

HYDE_PROMPT = """Bạn là một hệ thống tìm kiếm pháp luật thông minh (HyDE - Hypothetical Document Embeddings).
Nhiệm vụ của bạn là đọc các thông tin trích xuất từ một vụ việc thực tế, sau đó VIẾT LẠI thành một ĐOẠN VĂN BẢN HƯ CẤU giống hệt như văn phong của một điều luật hoặc một bản án mẫu.

**Yêu cầu:**
- Độ dài khoảng 3-4 câu.
- Dùng từ ngữ pháp lý chuẩn xác (ví dụ: người lao động, người sử dụng lao động, bồi thường thiệt hại ngoài hợp đồng, tài sản thừa kế...).
- Tập trung vào HÀNH VI, QUYỀN và NGHĨA VỤ pháp lý.
- KHÔNG giải thích, KHÔNG lặp lại tên người, KHÔNG phân tích vụ án. CHỈ viết ra đoạn văn bản giả định.

**Thông tin vụ việc:**
{query_text}

**Đoạn văn bản pháp luật giả định:**
"""

__all__ = [
    "RETRIEVE_LAW_PROMPT",
    "HYDE_PROMPT",
]
