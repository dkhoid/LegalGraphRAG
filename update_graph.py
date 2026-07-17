import os
import sys
import uuid
import numpy as np
import requests

sys.path.append(os.path.abspath("."))
from dotenv import load_dotenv

load_dotenv()

from core.graph_construct.graph_db import GraphDBManager


def get_embedding(text):
    url = "https://api.openai.com/v1/embeddings"
    model = "text-embedding-3-small"
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"model": model, "input": text}
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Failed to get embedding: {e}")
    return None


def main():
    graph_path = "./data/processed/graph.pkl"
    if not os.path.exists(graph_path):
        print("Graph not found")
        return

    GraphDBManager.load(graph_path)
    db = GraphDBManager.get_db()

    # Add the new case
    case_desc = "Lao động nữ sinh con cần chuẩn bị những hồ sơ gì để hưởng chế độ thai sản theo quy định mới nhất?"
    case_id = str(uuid.uuid4())
    case_emb = get_embedding(case_desc)

    db.add_node(
        case_id,
        "Cases",
        {
            "description": case_desc,
            "embedding": case_emb,
            "caseId": "qa_bhxh_1",
            "dispute": ["Bảo hiểm xã hội"],
            "law": ["Luật Bảo hiểm xã hội Điều 61"],
        },
    )

    # Add new laws
    laws = [
        {
            "id": "law_999999",
            "entry": 999999,
            "text": "Điều 101 Luật hôn nhân và gia đình năm 2014. Thẩm quyền giải quyết việc xác định cha, mẹ, con\n1. Cơ quan đăng ký hộ tịch có thẩm quyền xác định cha, mẹ, con theo quy định của pháp luật về hộ tịch trong trường hợp không có tranh chấp.\n2. Tòa án có thẩm quyền giải quyết việc xác định cha, mẹ, con trong trường hợp có tranh chấp hoặc người được yêu cầu xác định là cha, mẹ, con đã chết và trường hợp quy định tại Điều 92 của Luật này.\nQuyết định của Tòa án về xác định cha, mẹ, con phải được gửi cho cơ quan đăng ký hộ tịch để ghi chú theo quy định của pháp luật về hộ tịch; các bên trong quan hệ xác định cha, mẹ, con; cá nhân, cơ quan, tổ chức có liên quan theo quy định của pháp luật về tố tụng dân sự.",
            "dispute": ["Quy định chung"],
        },
        {
            "id": "law_1000000",
            "entry": 1000000,
            "text": "Điều 623 Bộ luật Dân sự 2015. Thời hiệu thừa kế\n1. Thời hiệu để người thừa kế yêu cầu chia di sản là 30 năm đối với bất động sản, 10 năm đối với động sản, kể từ thời điểm mở thừa kế. Hết thời hạn này thì di sản thuộc về người thừa kế đang quản lý di sản.\n2. Thời hiệu để người thừa kế yêu cầu xác nhận quyền thừa kế của mình hoặc bác bỏ quyền thừa kế của người khác là 10 năm, kể từ thời điểm mở thừa kế.\n3. Thời hiệu yêu cầu người thừa kế thực hiện nghĩa vụ về tài sản của người chết để lại là 03 năm, kể từ thời điểm mở thừa kế.",
            "dispute": ["Thừa kế", "Quyền sở hữu"],
        },
    ]

    for law in laws:
        emb = get_embedding(law["text"])
        db.add_node(
            law["id"],
            "Laws",
            {
                "entry": law["entry"],
                "description": law["text"],
                "embedding": emb,
                "disputes": law["dispute"],
                "judge_dep": [],
                "related_laws": [],
                "insights": "",
            },
        )

    GraphDBManager.save(graph_path)
    print("Graph updated successfully.")


if __name__ == "__main__":
    main()
