# Báo Cáo Tổng Quan Dự Án: LegalGraphRAG

## 1. Giới Thiệu Chung
**LegalGraphRAG** là một hệ thống phân tích Luật Dân sự Việt Nam tiên tiến, tích hợp công nghệ **Graph RAG** (Retrieval-Augmented Generation dựa trên Đồ thị Tri thức) đa tác tử. Hệ thống hỗ trợ người dùng đưa ra các hướng giải quyết pháp lý dựa trên các điều luật và án lệ/vụ án tương tự làm tài liệu tham khảo.

- **Lĩnh vực:** Legal Tech (Công nghệ Pháp lý)
- **Quốc gia:** Việt Nam (Tập trung vào Bộ luật Dân sự, Luật Lao động, và Luật Bảo hiểm Xã hội).

---

## 2. Các Tính Năng Nổi Bật
- ✅ **Đồ thị Tri thức Pháp luật (Knowledge Graph):** Kết nối các điều luật và các vụ án dân sự để tạo thành một mạng lưới tri thức phong phú, giúp cho việc truy xuất dữ liệu có tính ngữ cảnh cao hơn.
- ✅ **Truy xuất Tự động (Automated Retrieval):** Hệ thống có khả năng tìm kiếm các tình tiết vụ án tương tự và các điều luật liên quan trực tiếp từ đồ thị dựa trên thông tin vụ việc của người dùng (`fact`).
- ✅ **Phân tích Khởi tạo (Generative Analysis):** Sử dụng các mô hình Ngôn ngữ Lớn (LLMs) để phân tích các tranh chấp dân sự, xác định luật áp dụng, và đề xuất hướng giải quyết cụ thể bằng Tiếng Việt.
- ✅ **API Server & Giao diện Web:** Đi kèm một máy chủ FastAPI (`main.py`) và giao diện người dùng trực quan (`web/index.html`) giúp người dùng dễ dàng tương tác.

---

## 3. Kiến Trúc Hệ Thống (Architecture)
Dự án được tổ chức theo cấu trúc module hóa rất rõ ràng:

### 3.1. Thành phần Cốt lõi (`core/`)
- **`LegalGraphRAG.py`**: Trái tim của hệ thống. Quản lý việc nạp cấu hình (`LegalGraphRAGConfig`), khởi tạo mô hình AI, kết nối Graph DB và thực thi phân tích.
- **`models/`**: Hỗ trợ nhiều LLM khác nhau bao gồm cả mã nguồn mở và thương mại (ví dụ: Qwen, Gemma, InternLM, GLM, DeepSeek, GPT-4o-mini).
- **`graph_construct/`**: Chứa logic xây dựng Đồ thị Đặc trưng (Feature Graph) từ dữ liệu luật pháp và truy xuất các node tương tự.
- **`retriever/` & `judge/`**: Xử lý logic truy vấn và đánh giá kết quả.

### 3.2. Lớp Dịch Vụ API (`main.py`)
Sử dụng **FastAPI** để cung cấp các API endpoints phục vụ Web UI:
- `POST /analyze_civil`: Endpoint chính để phân tích vụ án, nhận input từ người dùng, gọi RAG và trả về kết quả JSON (bao gồm loại tranh chấp, luật áp dụng, và hướng giải quyết).
- `POST /generate_prompt`: Endpoint trung gian sinh ra prompt (bao gồm context từ luật và fact).
- `POST /api/chat`: Hỗ trợ giao tiếp dạng chat thông thường về vấn đề pháp lý.

### 3.3. Dữ Liệu và Tiện Ích (`scripts/`, `data/`)
- Các script Python để thu thập luật (`fetch_vn_legal_data.py`), sinh dữ liệu vụ án mẫu (`generate_sample_cases.py`), và trực quan hóa Đồ thị bằng Pyvis (`visualize_graph.py`).

---

## 4. Công Nghệ Sử Dụng (Tech Stack)
- **Backend & API:** Python, FastAPI, Uvicorn, Pydantic.
- **AI / Machine Learning:** PyTorch, Transformers, LangChain, OpenAI API.
- **Đồ thị / Dữ liệu:** Neo4j (được tích hợp cho Graph DB), NetworkX.
- **Frontend:** HTML, CSS, JavaScript (Vanilla).

---

## 5. Đánh Giá Hiện Trạng & Tiềm Năng
### Điểm Mạnh:
- **Linh hoạt trong AI:** Hỗ trợ nhiều mô hình LLM từ chạy cục bộ (Qwen) đến các API thương mại (DeepSeek, GPT), giúp cân bằng giữa hiệu suất và chi phí.
- **Phương pháp Graph RAG Tiên tiến:** Kết hợp RAG truyền thống với Graph giúp AI hiểu được mối quan hệ phức tạp giữa các điều luật Việt Nam thay vì chỉ tìm kiếm theo text similarity.
- **Cấu trúc dễ mở rộng:** Dễ dàng bổ sung thêm các bộ luật mới (như Luật Hình sự, Luật Thương mại) vào hệ thống.

### Điểm Cần Lưu Ý (Risks & Improvements):
- Prompt và LLM phụ thuộc nhiều vào khả năng tuân thủ định dạng (hiện tại `main.py` phải có logic fallback JSON).
- Việc xây dựng đồ thị ban đầu có thể tốn tài nguyên.

---
*Báo cáo được khởi tạo tự động bởi Antigravity Orchestrator (Bao gồm Explorer Agent & Project Planner).*
