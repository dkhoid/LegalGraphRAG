from typing import List, Dict, Any


def format_law(laws: List[Dict[str, Any]]) -> str:
    res = ""
    for law in laws:
        law_id = law.get("entry", law.get("id", "Unknown"))
        description = law.get("description", law.get("text", ""))
        res += f"Article {law_id}: {description}\n---\n"
    return res


def format_fact(facts: List[Dict[str, Any]]) -> str:
    res = ""
    for fact in facts:
        res += f"Similar Case Fact: {fact.get('description', '')}\n---\n"
    return res


def build_civil_prompt(
    fact: str, laws: List[Dict[str, Any]], similar_facts: List[Dict[str, Any]]
) -> str:
    prompt_template = """
Bạn là một chuyên gia tư vấn pháp lý chuyên về Luật Dân sự Việt Nam.
Dưới đây là các quy định pháp luật (điều luật) và các tiền lệ (án lệ/vụ án tương tự) được trích xuất từ hệ thống dữ liệu:

- Quy định pháp luật (Laws):
{formatted_laws}

- Án lệ/Vụ án tương tự (Similar Cases):
{formatted_facts}

**Tình tiết vụ án của người dùng cần phân tích:**
{fact}

**Hướng dẫn phân tích:**
1. Hãy đối chiếu tình tiết vụ án của người dùng với các dữ liệu được trích xuất ở trên.
2. Nếu các điều luật hoặc vụ án trích xuất **KHÔNG LIÊN QUAN** đến tình tiết của người dùng (ví dụ: người dùng hỏi về đòi nợ nhưng dữ liệu lại về hôn nhân/thai sản), hãy **BỎ QUA** dữ liệu trích xuất đó và tự suy luận dựa trên kiến thức pháp luật Dân sự Việt Nam của bạn (ví dụ: Bộ luật Dân sự về hợp đồng vay tài sản, nghĩa vụ trả nợ).
3. Xác định bản chất của quan hệ pháp luật và loại tranh chấp (VD: Tranh chấp hợp đồng vay tài sản, Bồi thường thiệt hại, v.v.).
4. Xác định các điều luật cụ thể áp dụng phù hợp nhất. Nếu dữ liệu cung cấp bị sai, hãy dùng kiến thức chuẩn xác của bạn để chỉ ra điều luật đúng. KHÔNG ĐƯỢC bịa ra số hiệu luật không tồn tại (như Điều 999999).
5. Đưa ra hướng giải quyết rõ ràng (VD: Ai là người chịu trách nhiệm? Có thể khởi kiện ở đâu? Thủ tục đòi nợ ra sao?).

YÊU CẦU QUAN TRỌNG: Toàn bộ câu trả lời PHẢI ĐƯỢC VIẾT BẰNG TIẾNG VIỆT.

**Định dạng đầu ra:**
Bạn CHỈ ĐƯỢC phép trả về một đối tượng JSON hợp lệ. Không bao gồm các khối mã markdown (như ```json), chỉ trả về JSON thuần túy.
Đối tượng JSON phải tuân thủ chính xác cấu trúc sau:
{{
    "dispute_type": "Tên loại tranh chấp dân sự",
    "applicable_laws": ["Điều X Luật Y", "Điều Z Luật W"],
    "resolution_direction": "Giải thích chi tiết về hướng giải quyết pháp lý và trách nhiệm (BẰNG TIẾNG VIỆT)"
}}
"""
    # Xử lý sanitize input đơn giản
    safe_fact = fact.replace("{", "{{").replace("}", "}}")
    return prompt_template.format(
        formatted_laws=format_law(laws).replace("{", "{{").replace("}", "}}"),
        formatted_facts=format_fact(similar_facts).replace("{", "{{").replace("}", "}}"),
        fact=safe_fact,
    )
