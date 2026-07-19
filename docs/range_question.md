Listed directory processed
Listed directory prompt
Viewed __init__.py:1-82
Viewed __init__.py:1-136

🤖 **Applying knowledge of `@[backend-specialist]` & `@[explorer-agent]`...**

Dựa trên cấu trúc dữ liệu, đồ thị tri thức và các câu lệnh prompt (hướng dẫn cho AI) được thiết lập trong mã nguồn hệ thống LegalGraphRAG của bạn, **độ rộng và phạm vi câu hỏi mà RAG này có thể trả lời tốt nhất** nằm trong khoảng sau đây:

### 1. Phạm vi về Lĩnh vực Pháp lý (Domain Scope)
Hệ thống được thiết kế đặc thù cho **Luật Dân sự Việt Nam** (như đã định nghĩa trong `README.md` và `law_to_dispute.json`), bao gồm mở rộng sang một số lĩnh vực có liên quan chặt chẽ:
- **Dân sự cốt lõi:** Tranh chấp hợp đồng, thừa kế, quyền sở hữu tài sản, bồi thường thiệt hại ngoài hợp đồng, hôn nhân và gia đình.
- **Lao động & Bảo hiểm (nếu có nạp dữ liệu):** Tranh chấp lao động, sa thải trái pháp luật, quyền lợi bảo hiểm.
- ❌ **Ngoài phạm vi:** Không tối ưu cho Luật Hình sự, Luật Hành chính, Luật Doanh nghiệp (trừ khi liên quan trực tiếp đến tranh chấp dân sự).

### 2. Định dạng Câu hỏi Tối ưu (Input Scope)
Hệ thống **KHÔNG** phải là một chatbot hỏi đáp lý thuyết thông thường. Nó là một cỗ máy phân tích vụ việc (Legal Reasoning).
- **✅ Câu hỏi NẰM TRONG phạm vi (Case-based / Tình tiết thực tế):** Hệ thống yêu cầu đầu vào là một `fact` (Mô tả tình tiết vụ án).
  - *Ví dụ tốt:* "A cho B vay 500 triệu đồng có viết giấy tay, hẹn 3 tháng trả nhưng quá hạn B không trả và bỏ trốn. Xin hỏi A phải làm gì?"
  - *Ví dụ tốt:* "Công ty X đơn phương chấm dứt hợp đồng lao động với nhân viên Y đang mang thai. Y có quyền lợi gì?"
- **❌ Câu hỏi NẰM NGOÀI phạm vi (Lý thuyết / Trừu tượng):** Các câu hỏi không có tình huống thực tế cụ thể hoặc mang tính chất lý luận hàn lâm sẽ khiến RAG bị bối rối vì prompt của nó luôn cố gắng bóc tách "hành vi", "đương sự".
  - *Ví dụ xấu:* "Luật Dân sự năm 2015 có bao nhiêu chương và điều?" (AI sẽ cố gắng tìm "đương sự" trong câu này và thất bại).
  - *Ví dụ xấu:* "Khái niệm năng lực hành vi dân sự là gì?"

### 3. Phạm vi Câu trả lời / Đầu ra (Output Scope)
Khi nhận được câu hỏi đúng chuẩn (tình tiết vụ việc), RAG sẽ xử lý qua các hàm đánh giá (`JUDGE_CIVIL_ALL_PROMPT`) và trả lời chính xác trong 3 khía cạnh:
1. **Phân loại tranh chấp (Dispute Type):** Xác định bản chất pháp lý của vụ việc (Ví dụ: Tranh chấp hợp đồng vay tài sản).
2. **Xác định Luật áp dụng (Applicable Laws / Law Articles):** Trích xuất chính xác Điều, Khoản liên quan trong bộ luật điều chỉnh hành vi đó.
3. **Đề xuất Hướng giải quyết (Resolution Direction):** Đưa ra lời giải pháp lý:
   - **Trách nhiệm dân sự (Liability):** Ai là người chịu trách nhiệm, ai đúng ai sai.
   - **Bồi thường/Xử lý (Compensation):** Hình thức khắc phục hậu quả, xử lý tài sản dựa theo thực tiễn xét xử (tiền lệ/án lệ được nạp vào mạng lưới).

### Tóm lại
**Câu hỏi của bạn sẽ nằm trong phần trả lời hoàn hảo của RAG nếu nó là:**
> *Một đoạn mô tả tình huống tranh chấp thực tế giữa các bên trong khuôn khổ luật dân sự/lao động, với mong muốn biết ai đúng ai sai, cần áp dụng điều luật nào và hướng giải quyết ra sao.*
