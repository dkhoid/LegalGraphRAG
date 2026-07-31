import json
import os
import shutil
import re


def main():
    cases_path = "data/processed/cases_with_feature.json"
    backup_path = cases_path + ".law_bak"

    if not os.path.exists(backup_path):
        print(f"Creating backup at {backup_path}")
        shutil.copy2(cases_path, backup_path)

    print("Loading cases...")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Law prefixes dictionary
    # Map common law text -> Zalo prefix
    law_mapping = {
        "bộ luật dân sự 2015": "zalo_91/2015/qh13",
        "bộ luật hình sự": "zalo_100/2015/qh13",  # Assuming 2015/2017 consolidated
        "bộ luật tố tụng dân sự 2015": "zalo_92/2015/qh13",
        "bộ luật tố tụng hình sự 2015": "zalo_101/2015/qh13",
        "bộ luật lao động 2019": "zalo_45/2019/qh14",
        "luật hôn nhân và gia đình 2014": "zalo_52/2014/qh13",
        "luật doanh nghiệp 2020": "zalo_59/2020/qh14",
        "luật đất đai 2013": "zalo_45/2013/qh13",
        "luật thương mại 2005": "zalo_36/2005/qh11",
        "luật phá sản 2014": "zalo_51/2014/qh13",
        "luật nhà ở 2014": "zalo_65/2014/qh13",
        "luật bảo vệ quyền lợi người tiêu dùng 2010": "zalo_59/2010/qh12",
        "luật sở hữu trí tuệ 2005": "zalo_50/2005/qh11",
    }

    # Regex to extract "Điều X"
    # Matches "Điều 581", "khoản 1 Điều 581", "Điều 43" etc.
    dieu_pattern = re.compile(r"điều\s+(\d+)", re.IGNORECASE)

    updated_count = 0
    not_matched_count = 0
    total_laws_checked = 0

    for c in cases:
        laws = c.get("law", [])
        if not laws:
            continue

        new_laws = []
        for law in laws:
            law_str = str(law).strip().lower()
            if not law_str:
                continue
            total_laws_checked += 1

            # If it's already a zalo format or simple number, keep it
            if law_str.startswith("zalo_") or law_str.isdigit():
                new_laws.append(law)
                continue

            # Otherwise, try to map it
            matched = False
            for key, prefix in law_mapping.items():
                if key in law_str:
                    # Extract number
                    match = dieu_pattern.search(law_str)
                    if match:
                        dieu_num = match.group(1)
                        new_law_id = f"{prefix}+{dieu_num}"
                        new_laws.append(new_law_id)
                        matched = True
                        updated_count += 1
                        break

            if not matched:
                new_laws.append(law)  # Keep original if we can't map
                not_matched_count += 1

        # Remove duplicates
        c["law"] = list(dict.fromkeys(new_laws))

    print(f"Total law strings checked: {total_laws_checked}")
    print(f"Successfully normalized to Zalo format: {updated_count}")
    print(f"Could not normalize: {not_matched_count}")

    print("Saving normalized cases...")
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=4)

    print("Done! Law ID normalization complete.")


if __name__ == "__main__":
    main()
