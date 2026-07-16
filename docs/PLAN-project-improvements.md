# Plan: Project Improvements

Dựa trên các khuyến nghị từ báo cáo đánh giá dự án (Project Review), bản kế hoạch này được lập ra nhằm nâng cấp hệ thống LegalGraphRAG để đạt tiêu chuẩn Production-ready.

---

## 🛑 Socratic Gate (Open Questions)
Trước khi bắt đầu triển khai `/create`, xin vui lòng xác nhận các thông tin sau:
1. **Neo4j Deployment:** Bạn muốn cài đặt Neo4j ở dưới dạng Local (Docker) hay sử dụng dịch vụ đám mây (Neo4j AuraDB)?
2. **Logging Framework:** Cấu hình log sẽ lưu ra file `.log` hay chỉ in ra terminal để hệ thống quản lý container thu thập?
3. **CI/CD:** Bạn có muốn tích hợp luôn pre-commit hooks (chạy lint/test trước khi commit) ngoài việc tạo workflow GitHub Actions không?

---

## 🎯 Task Breakdown

### Phase 1: CI/CD & Testing Infrastructure
- **Agent:** `test-engineer`, `backend-specialist`
- **Nhiệm vụ:**
  - Bổ sung `pytest`, `pytest-cov` vào `requirements.txt`.
  - Cấu hình file `pytest.ini` cơ bản.
  - Viết file workflow `.github/workflows/python-app.yml` để chạy test tự động khi push/pull_request.

### Phase 2: System Logging
- **Agent:** `backend-specialist`, `clean-code`
- **Nhiệm vụ:**
  - Khởi tạo module `core/utils/logger.py` cấu hình chuẩn logging của Python (hỗ trợ format thời gian, file/console handler).
  - Thay thế toàn bộ các lệnh `print()` trong `core/LegalGraphRAG.py`, `api_server.py`, `scripts/*` bằng `logger.info()`, `logger.error()`, v.v.

### Phase 3: Graph Scalability (Neo4j Integration)
- **Agent:** `database-architect`, `backend-specialist`
- **Nhiệm vụ:**
  - Thêm `neo4j` vào `requirements.txt`.
  - Cập nhật `.env` và `env.example` thêm các biến `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
  - Viết module kết nối và xử lý dữ liệu Graph qua Neo4j trong `core/graph_construct/graph_db.py` (hoặc tạo file mới `neo4j_manager.py`).
  - Nâng cấp logic ở `feature_graph.py` để ghi dữ liệu node/edge trực tiếp vào Neo4j thay vì dùng memory/pkl.

### Phase 4: Verification & Security
- **Agent:** `security-auditor`
- **Nhiệm vụ:**
  - Đảm bảo `.env` được setup chuẩn trong `.gitignore`.
  - Chạy `python .agent/scripts/security_scan.py .` để rà soát mã nguồn.
  - Chạy `pytest` để xác minh các component không bị ảnh hưởng sau khi thay đổi log và Graph DB.

---

## ✅ Verification Checklist (Phase X)
- [ ] Pytest chạy thành công qua lệnh `python -m pytest tests/`.
- [ ] Log ghi nhận được lịch sử kết nối Graph và phân tích vụ án mà không còn dòng `print` nào trong console.
- [ ] Graph hiển thị chính xác trên Browser của Neo4j và giao diện Web UI cũ vẫn có thể fetch/query từ hệ thống.
