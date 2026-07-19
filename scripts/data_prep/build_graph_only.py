import os
import sys

# Thêm đường dẫn thư mục gốc vào sys.path để import các module core
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.LegalGraphRAG import LegalGraphRAG, LegalGraphRAGConfig
from core.utils.logger import logger


def main():
    logger.info("=" * 60)
    logger.info("Bắt đầu quá trình tạo Graph độc lập...")
    logger.info("=" * 60)

    # Đọc cấu hình từ file .env
    config = LegalGraphRAGConfig.from_env_file(".env")

    # Khởi tạo RAG builder
    rag_builder = LegalGraphRAG(config=config)

    # Build graph (Bắt buộc tạo lại mới với cờ force_rebuild=True)
    rag_builder.build_graph(force_rebuild=True)

    # Giải phóng model sau khi tạo xong
    if hasattr(rag_builder, "model") and hasattr(rag_builder.model, "release_model"):
        try:
            rag_builder.model.release_model()
        except Exception:
            pass

    logger.info("=" * 60)
    logger.info("Tạo Graph thành công!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
