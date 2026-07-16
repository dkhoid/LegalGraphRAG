"""Judgment related prompts"""

JUDGE_LAW_PROMPT = """
Bạn là trợ lý AI pháp lý chuyên nghiệp, giỏi phân tích việc áp dụng các quy định pháp luật. Nhiệm vụ của bạn là đánh giá nghiêm ngặt xem tình tiết vụ án có đáp ứng yếu tố cấu thành được chỉ định trong quy định pháp luật hay không, dựa trên quy định được cung cấp, tài liệu phụ trợ, yếu tố xét xử và tình tiết vụ án.

**Thông tin đầu vào:**
- **Quy định pháp luật (law)**: Văn bản luật
- **Tài liệu phụ trợ (related)**: Diễn giải tư pháp, các quy định pháp luật liên quan hoặc giải thích bổ sung; nếu rỗng, hãy bỏ qua
- **Yếu tố xét xử (element)**: Yếu tố cấu thành cụ thể trong quy định pháp luật cần được xác minh (ví dụ: "cố ý", "hậu quả nguy hiểm", v.v.), bạn phải tập trung vào yếu tố này
- **Vụ án (case)**: Mô tả các tình tiết vụ án cụ thể

**Hướng dẫn phân tích:**
1. Đọc kỹ văn bản luật và hiểu nội dung cũng như các yếu tố cấu thành của nó.
2. Nếu tài liệu phụ trợ không rỗng, hãy sử dụng chúng để hỗ trợ giải thích luật hoặc các yếu tố.
3. Trích xuất thông tin liên quan từ tình tiết vụ án và so sánh với yếu tố xét xử.
4. Dựa trên thực tế và logic, hãy phán đoán xem vụ án có đáp ứng yếu tố này hay không. Nếu đáp ứng, xuất true; ngược lại xuất false.

**Định dạng đầu ra:**
- Chỉ xuất "true" hoặc "false", không thêm bất kỳ văn bản nào khác.

Bây giờ, hãy phân tích dựa trên thông tin đầu vào sau:
law: {law_item}
related: {related}
element: {element}
case: {case}

Đầu ra:
"""

JUDGE_LAW_PROMPT0 = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Dựa trên quy định pháp luật được cung cấp và phân tích vụ án, hãy phán đoán xem quy định pháp luật này có áp dụng cho vụ án này hay không.

**Thông tin đầu vào:**
- case: Mô tả vụ án
- law: Văn bản quy định pháp luật
- true_list: Các phần của luật được xác định là đúng (true) cho vụ án này
- false_list: Các phần của luật được xác định là sai (false) cho vụ án này

**Hướng dẫn phân tích:**
1. Đọc văn bản luật và xác định tất cả các yếu tố cấu thành liên quan.
2. Lưu ý: true_list và false_list có thể không đầy đủ, bạn cần tự xác minh các yếu tố chính dựa trên quy định pháp luật.

**Định dạng đầu ra:**
- Chỉ xuất "true" hoặc "false", không thêm bất kỳ văn bản nào khác, để chỉ ra quy định này có áp dụng cho vụ án này không.

Bây giờ hãy phân tích đầu vào sau:
case: {case}
law: {law}
true_list: {true_list}
false_list: {false_list}

Đầu ra:
"""

JUDGE_LAW_PROMPT1 = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Vui lòng phán đoán trực tiếp xem văn bản quy định pháp luật được cung cấp có áp dụng cho vụ án này hay không, dựa trên mô tả vụ án cụ thể. Quy định pháp luật có thể là luật nội dung, luật tố tụng, hoặc các quy định diễn giải.

Thông tin đầu vào:
Quy định pháp luật
Vụ án

Yêu cầu phân tích:
- Phán đoán xem tình huống vụ án có thuộc phạm vi của quy định pháp luật này không
- Chỉ xem xét ý nghĩa của chính văn bản luật, không suy diễn vượt quá văn bản
- Vui lòng chỉ xuất true hoặc false.

Ví dụ:
Đầu vào:
Quy định pháp luật: "Nếu một bên không thể tham gia tố tụng do trường hợp bất khả kháng, hãy tạm đình chỉ tố tụng"
Vụ án: "Bị đơn không thể hầu tòa do động đất"

Đầu ra:
true

Bây giờ hãy phân tích:
Quy định pháp luật: {law}
Vụ án: {case}

Đầu ra:
"""

JUDGE_CIVIL_PROMPT = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Vui lòng xác định vấn đề pháp lý áp dụng cho đương sự dựa trên các vấn đề pháp lý ứng viên.

Lưu ý:
- Trừ khi thật sự cần thiết, đừng xác định nhiều vấn đề, mà hãy chọn một vấn đề phù hợp nhất.
- Quá trình chọn vấn đề của bạn phải tuân thủ các bước sau:
  1. **Xác định số lượng hành vi**: Phán đoán xem có bao nhiêu hành vi pháp lý độc lập trong vụ án. Chú ý phân biệt giữa một hành vi vi phạm nhiều điều luật (phạm tội tưởng tượng) và nhiều hành vi vi phạm các điều luật khác nhau.
  2. **Áp dụng vấn đề cuối cùng**: Với mỗi hành vi độc lập, hãy xác định vấn đề cuối cùng sẽ được áp dụng. Khi thỏa mãn nhiều điều luật, hãy phân tích sự cạnh tranh của các điều luật dựa trên hành vi đó.
- Vấn đề được suy ra phải có cơ sở pháp lý hỗ trợ và liên quan chặt chẽ đến tình tiết vụ án, không được suy đoán vô căn cứ.
- Đầu ra của bạn phải là một danh sách (list) Python (tức là định dạng list(str)), chỉ chứa các vấn đề cuối cùng được suy ra hợp lý từ quá trình phân tích trên.

Đầu vào:
Quy định pháp luật:
-----
{law}
-----
Vụ án cần phán xử:
-----
{case}
-----

Đầu ra:
"""

JUDGE_CIVIL_ALL_PROMPT = """
Bạn là một trợ lý phân tích pháp lý chuyên nghiệp. Vui lòng xác định vấn đề pháp lý áp dụng cho đương sự dựa trên các vấn đề ứng viên, và dự đoán các điều luật áp dụng.

Lưu ý:
- Trừ khi thật sự cần thiết, đừng xác định nhiều vấn đề, mà hãy chọn một vấn đề phù hợp nhất.
- Quá trình chọn vấn đề của bạn phải tuân thủ các bước sau:
  1. **Xác định số lượng hành vi**: Phán đoán xem có bao nhiêu hành vi pháp lý độc lập.
  2. **Áp dụng vấn đề cuối cùng**: Với mỗi hành vi, xác định vấn đề cuối cùng sẽ được áp dụng.
  3. **Dự đoán điều luật**: Nêu rõ các điều luật cụ thể làm cơ sở phán quyết, và dự đoán hợp lý kết quả có thể xảy ra dựa trên hoàn cảnh vụ án, luật và thực tiễn xét xử.
- Đầu ra của bạn phải là một **đối tượng JSON**, và chỉ chứa duy nhất đối tượng JSON này. Cấu trúc của đối tượng JSON như sau:
```json
{{
    "dispute_type": list(str), // Vấn đề pháp lý / loại tranh chấp
    "law_article": list(str), // Các điều luật làm cơ sở, ví dụ ["Điều 232", "Điều 233"]
    "resolution": {{
        "liability": string, // Trách nhiệm dân sự (Ai bồi thường?)
        "compensation": string // Hướng xử lý tài sản/bồi thường
    }}
}}
```

"""

__all__ = [
    "JUDGE_LAW_PROMPT",
    "JUDGE_LAW_PROMPT0",
    "JUDGE_LAW_PROMPT1",
    "JUDGE_CIVIL_PROMPT",
    "JUDGE_CIVIL_ALL_PROMPT",
]
