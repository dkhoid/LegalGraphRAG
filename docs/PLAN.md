# Dự án: Tối ưu hóa và Đưa vào thực tiễn (Production-Ready)

Dựa trên yêu cầu của bạn, chúng ta sẽ quay trở lại kế hoạch ban đầu, tập trung vào việc Đóng gói (Docker) và Vận hành để dự án sẵn sàng deploy.

## Đánh giá Tình trạng Hiện tại (Current State)
1. **Cấu trúc đã ổn định**: Thư mục gốc đã được dọn dẹp sạch sẽ, code được chia module khá tốt (`core`, `scripts`, `tests`, `web`).
2. **Thiếu Docker hóa toàn diện**: Hiện tại `docker-compose.yml` mới chỉ có duy nhất database Neo4j. Backend chính (`main.py`) vẫn phải chạy chay bằng lệnh ở máy local.
3. **Thiếu Unit Tests**: Có thư mục `tests/` nhưng chỉ có 2 file test rất mỏng (`test_neo4j.py`, `test_recovery.py`).
4. **Rủi ro bảo mật nhỏ**: Cần quét lại mã nguồn để đảm bảo không bị rò rỉ API key hoặc cấu hình CORS.

---

## Các Hạng Mục Công Việc Ưu Tiên (Priority Tasks)

Dưới đây là kế hoạch chi tiết cho các Agent thực hiện song song (Phase 2) nếu bạn phê duyệt:

### Nhóm 1: Triển khai & Vận hành (Foundation - `devops-engineer`)
*Mục tiêu: Đóng gói dự án để có thể chạy ở bất kỳ đâu chỉ với 1 lệnh.*
- Tạo file `Dockerfile` tối ưu hóa (multi-stage build) cho backend FastAPI.
- Cập nhật `docker-compose.yml` để chạy đồng thời cả `backend` (FastAPI) và `neo4j` cùng nhau trong chung một mạng lưới (network).

### Nhóm 2: Đảm bảo Chất lượng (Core - `test-engineer`)
*Mục tiêu: Tạo lớp bảo vệ để tránh lỗi khi cập nhật code sau này.*
- Cấu hình thư viện `pytest` chuẩn mực.
- Viết thêm Unit Test để đảm bảo luồng API (endpoints trong `main.py`) hoạt động trơn tru.

### Nhóm 3: Bảo mật (Polish - `security-auditor`)
*Mục tiêu: Đảm bảo không có lỗ hổng rò rỉ dữ liệu hoặc chèn mã độc (Injection).*
- Quét toàn bộ `main.py` để vá các cấu hình CORS lỏng lẻo.
- Xóa các token bị lộ hoặc sử dụng biến môi trường `.env` triệt để.

---

## Phê duyệt (User Approval Required)

> [!IMPORTANT]
> Đây là kế hoạch triển khai (Deploy) ban đầu.
>
> Nếu bạn đồng ý với kế hoạch này (nhấn **Proceed**), hệ thống sẽ chuyển sang Phase 2: Kích hoạt đồng thời 3 Agent (`devops-engineer`, `test-engineer`, `security-auditor`) để thực thi Docker hóa và test toàn hệ thống.
