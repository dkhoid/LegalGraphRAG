# Sử dụng Python 3.11 slim cho image nhẹ
FROM python:3.11-slim as builder

# Đặt thư mục làm việc
WORKDIR /app

# Thiết lập các biến môi trường để Python chạy mượt hơn trong Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Cài đặt các gói hệ thống cần thiết (nếu có thư viện C cần compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện Python
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# --- Stage 2: Final Image ---
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Chỉ copy các thư viện đã cài đặt từ builder stage sang để giảm dung lượng
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy toàn bộ mã nguồn vào container
COPY . .

# Expose cổng 8000 cho FastAPI
EXPOSE 8000

# Lệnh khởi chạy server bằng Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
