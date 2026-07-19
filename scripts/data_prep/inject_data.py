import json

# 1. Update cases_with_feature.json
with open("data/processed/cases_with_feature.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

# Add a Social Insurance test case
new_case = {
    "id": "qa_bhxh_1",
    "fact": "Lao động nữ sinh con cần chuẩn bị những hồ sơ gì để hưởng chế độ thai sản theo quy định mới nhất?",
    "dispute": ["Bảo hiểm xã hội"],
    "law": ["Luật Bảo hiểm xã hội Điều 61"],
    "laws": ["Luật Bảo hiểm xã hội Điều 61"],
    "domain": "bảo hiểm xã hội",
}
cases.insert(0, new_case)

with open("data/processed/cases_with_feature.json", "w", encoding="utf-8") as f:
    json.dump(cases, f, ensure_ascii=False, indent=2)


# 2. Inject Civil/Marriage Law into law_to_dispute.json
with open("data/processed/law_to_dispute.json", "r", encoding="utf-8") as f:
    laws_db = json.load(f)

new_law_id = 999999

dieu_101 = {
    "id": new_law_id,
    "items": [
        {
            "text": "Điều 101 Luật hôn nhân và gia đình năm 2014. Thẩm quyền giải quyết việc xác định cha, mẹ, con\n1. Cơ quan đăng ký hộ tịch có thẩm quyền xác định cha, mẹ, con theo quy định của pháp luật về hộ tịch trong trường hợp không có tranh chấp.\n2. Tòa án có thẩm quyền giải quyết việc xác định cha, mẹ, con trong trường hợp có tranh chấp hoặc người được yêu cầu xác định là cha, mẹ, con đã chết và trường hợp quy định tại Điều 92 của Luật này.\nQuyết định của Tòa án về xác định cha, mẹ, con phải được gửi cho cơ quan đăng ký hộ tịch để ghi chú theo quy định của pháp luật về hộ tịch; các bên trong quan hệ xác định cha, mẹ, con; cá nhân, cơ quan, tổ chức có liên quan theo quy định của pháp luật về tố tụng dân sự.",
            "dispute": ["Quy định chung"],
            "judge_dep": [],
            "related_laws": [],
        }
    ],
}

dieu_623 = {
    "id": new_law_id + 1,
    "items": [
        {
            "text": "Điều 623 Bộ luật Dân sự 2015. Thời hiệu thừa kế\n1. Thời hiệu để người thừa kế yêu cầu chia di sản là 30 năm đối với bất động sản, 10 năm đối với động sản, kể từ thời điểm mở thừa kế. Hết thời hạn này thì di sản thuộc về người thừa kế đang quản lý di sản.\n2. Thời hiệu để người thừa kế yêu cầu xác nhận quyền thừa kế của mình hoặc bác bỏ quyền thừa kế của người khác là 10 năm, kể từ thời điểm mở thừa kế.\n3. Thời hiệu yêu cầu người thừa kế thực hiện nghĩa vụ về tài sản của người chết để lại là 03 năm, kể từ thời điểm mở thừa kế.",
            "dispute": ["Thừa kế", "Quyền sở hữu"],
            "judge_dep": [],
            "related_laws": [],
        }
    ],
}

laws_db.extend([dieu_101, dieu_623])

with open("data/processed/law_to_dispute.json", "w", encoding="utf-8") as f:
    json.dump(laws_db, f, ensure_ascii=False, indent=2)

print("Injected test case and missing laws successfully!")
