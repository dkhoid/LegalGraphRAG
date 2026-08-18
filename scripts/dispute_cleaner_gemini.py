import json
import time
import os
import logging
from tqdm import tqdm
import requests
from dotenv import load_dotenv
import sys

# ==========================================
# Cấu hình Logging & Taxonomy
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dispute_cleaner.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

TAXONOMY = [
    "Quyền sở hữu",
    "Hợp đồng",
    "Bồi thường thiệt hại",
    "Thừa kế",
    "Hợp đồng thuê",
    "Tranh chấp lao động",
    "Nghĩa vụ dân sự",
    "Quy định chung",
    "Hợp đồng mua bán",
    "Sa thải, chấm dứt HĐLĐ",
    "Giao dịch dân sự",
    "Thế chấp, cầm cố",
    "Đại diện",
    "Hợp đồng vay tài sản",
    "Tiền lương",
    "Hợp đồng lao động",
    "Hợp đồng lao vụ",
    "An toàn lao động",
    "Bảo hiểm thất nghiệp",
    "Thời giờ làm việc",
    "Chế độ thai sản",
    "Bảo hiểm y tế",
]


# ==========================================
# Hàm gọi Gemini API
# ==========================================
def call_gemini(batch_data, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"

    prompt_text = "Dưới đây là danh sách các điều luật với ID tương ứng. Hãy phân loại từng điều luật vào 1 hoặc TỐI ĐA 2 chủ đề phù hợp nhất từ DANH SÁCH CHỦ ĐỀ CHUẨN. BẮT BUỘC chỉ trả về định dạng JSON dictionary, với key là ID của luật và value là mảng tên chủ đề.\n\n"
    prompt_text += f"DANH SÁCH CHỦ ĐỀ CHUẨN: {', '.join(TAXONOMY)}\n\n"
    prompt_text += 'Ví dụ định dạng trả về:\n```json\n{\n  "0": ["Hợp đồng"],\n  "1": ["Thừa kế", "Quyền sở hữu"]\n}\n```\n\nDanh sách luật cần phân loại:\n'

    for law_id, text in batch_data.items():
        prompt_text += f"ID: {law_id} | Nội dung: {text[:1000]}...\n---\n"

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"temperature": 0.0, "response_mime_type": "application/json"},
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            res_json = response.json()
            if "candidates" in res_json and len(res_json["candidates"]) > 0:
                text_response = res_json["candidates"][0]["content"]["parts"][0]["text"]
                text_response = (
                    text_response.strip().removeprefix("```json").removesuffix("```").strip()
                )
                try:
                    return json.loads(text_response)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {text_response}")
                    return None
            else:
                logger.error("No candidates returned from Gemini.")
                return None
        elif response.status_code == 429:
            logger.warning("Rate limit exceeded! (429)")
            return "429"
        else:
            logger.error(f"API Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Request Exception: {e}")
        return None


# ==========================================
# Luồng Xử lý Chính (Có Checkpoint Resume)
# ==========================================
def main():
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        logger.error("❌ Không tìm thấy GEMINI_API_KEY trong file .env.")
        return

    input_file = "data/clean/law_to_dispute_clean.json"
    output_file = "data/clean/law_to_dispute_normalized.json"
    checkpoint_file = "data/clean/checkpoint.txt"

    start_index = 0

    # Kịch bản 1: Đang chạy dở dang (Có file output và checkpoint)
    if os.path.exists(output_file) and os.path.exists(checkpoint_file):
        logger.info(f"🔄 Phát hiện tiến trình đang dang dở. Tiếp tục từ file: {output_file}")
        with open(checkpoint_file, "r") as f:
            try:
                start_index = int(f.read().strip())
                logger.info(f"▶️ Sẽ tiếp tục chạy từ batch số: {start_index}")
            except Exception:
                start_index = 0

        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Kịch bản 2: Chạy lần đầu tiên
    else:
        if not os.path.exists(input_file):
            logger.error(f"Không tìm thấy file data: {input_file}")
            return

        logger.info("▶️ Chạy lần đầu tiên. Nạp dữ liệu gốc...")
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Tạo danh sách các item cần xử lý
    # Lưu ý: items_to_process lưu tham chiếu (reference) đến dict gốc trong data
    items_to_process = []
    for doc in data:
        for item in doc.get("items", []):
            if "text" in item and item["text"].strip():
                items_to_process.append(item)

    total_items = len(items_to_process)
    logger.info(f"Tổng số lượng điều luật: {total_items}")

    BATCH_SIZE = 20

    def save_checkpoint(current_idx):
        # Lưu file data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Lưu con trỏ tiến trình
        with open(checkpoint_file, "w") as f:
            f.write(str(current_idx))
        logger.info(f"💾 Đã lưu Checkpoint tại index {current_idx}")

    try:
        # Nhảy đến vị trí start_index để tiếp tục
        for i in tqdm(
            range(start_index, total_items, BATCH_SIZE),
            desc="Tiến trình",
            initial=start_index // BATCH_SIZE,
            total=total_items // BATCH_SIZE,
        ):
            batch = items_to_process[i : i + BATCH_SIZE]
            batch_dict = {str(idx): item["text"] for idx, item in enumerate(batch)}

            retries = 3
            while retries > 0:
                result = call_gemini(batch_dict, GEMINI_API_KEY)
                if result == "429":
                    logger.warning("Đụng Rate Limit! Tạm nghỉ 15 giây...")
                    time.sleep(15)
                    retries -= 1
                    continue
                elif result is not None:
                    for idx_str, disputes in result.items():
                        idx = int(idx_str)
                        if idx < len(batch):
                            clean_disputes = [d for d in disputes if d in TAXONOMY]
                            if not clean_disputes:
                                clean_disputes = disputes
                            batch[idx]["dispute"] = clean_disputes
                    break
                else:
                    logger.error("Lỗi API, thử lại...")
                    time.sleep(5)
                    retries -= 1

            if retries == 0:
                logger.error(
                    f"❌ Batch tại index {i} thất bại hoàn toàn. Dừng chương trình để bảo toàn data."
                )
                save_checkpoint(i)
                sys.exit(1)

            time.sleep(5)  # Throttling

            # Save định kỳ mỗi 50 batches (1000 luật)
            if i > start_index and (i // BATCH_SIZE) % 50 == 0:
                save_checkpoint(i + BATCH_SIZE)

        # Hoàn thành toàn bộ
        save_checkpoint(total_items)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)  # Xóa checkpoint khi đã xong 100%
        logger.info(f"🎉 Hoàn thành xuất sắc! Dữ liệu được chốt tại: {output_file}")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Người dùng buộc dừng chương trình (Ctrl+C)!")
        logger.info("Đang tiến hành lưu lại dữ liệu (Khẩn cấp)...")
        # i là biến vòng lặp hiện tại, nếu đang ở giữa chừng, ta save lại vị trí i
        save_checkpoint(i)
        logger.info("✅ Đã lưu an toàn. Bạn có thể chạy lại lệnh để tiếp tục từ đây.")
        sys.exit(0)


if __name__ == "__main__":
    main()
