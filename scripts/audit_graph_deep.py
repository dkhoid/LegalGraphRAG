"""
Graph Deep Audit - Phần 2: Kiểm tra sâu các vấn đề đã phát hiện
"""

import pickle
import sys
import os
import random
import json
import numpy as np
from collections import Counter, defaultdict

PROJECT_ROOT = "/home/rokisaki/Documents/Coding/testing_code/LegalGraphRAG"
GRAPH_PATH = os.path.join(PROJECT_ROOT, "data/processed/graph.pkl")


def load_graph(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def check_dispute_names(data):
    """Check if Dispute nodes actually have names"""
    nodes_data = data["nodes_data"]

    print("=" * 70)
    print("1. KIEM TRA TEN CUA DISPUTE NODES")
    print("=" * 70)

    disputes = {nid: ninfo for nid, ninfo in nodes_data.items() if ninfo["type"] == "Disputes"}
    print(f"\nTong dispute nodes: {len(disputes)}")
    print("\nChi tiet tung dispute node:")

    for did, dinfo in sorted(disputes.items()):
        d = dinfo["data"]
        print(f"\n  ID: {did}")
        print(f"  Keys: {list(d.keys())}")
        for k, v in d.items():
            if k != "embedding":
                val_str = str(v)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                print(f"    {k}: {val_str}")


def check_law_dispute_field_empty(data):
    """Check how many laws have empty dispute field"""
    nodes_data = data["nodes_data"]
    graph = data["graph"]

    print("\n" + "=" * 70)
    print("2. KIEM TRA LAWS CO DISPUTE FIELD RONG")
    print("=" * 70)

    laws = {nid: ninfo for nid, ninfo in nodes_data.items() if ninfo["type"] == "Laws"}

    empty_dispute = 0
    has_dispute = 0
    for lid, linfo in laws.items():
        dispute_field = linfo["data"].get("dispute", [])
        if not dispute_field:
            empty_dispute += 1
        else:
            has_dispute += 1

    print(f"\nTong laws: {len(laws)}")
    print(f"Laws co dispute field: {has_dispute} ({100*has_dispute/len(laws):.1f}%)")
    print(f"Laws KHONG co dispute field: {empty_dispute} ({100*empty_dispute/len(laws):.1f}%)")

    # Of those with empty dispute field, how many still have RELATED_DISPUTE edges?
    empty_with_edges = 0
    for u, v, d in graph.edges(data=True):
        if d.get("relation_type") == "RELATED_DISPUTE":
            law_data = nodes_data.get(u, {}).get("data", {})
            if not law_data.get("dispute", []):
                empty_with_edges += 1

    print(f"\nLaws KHONG co dispute field NHUNG VAN co RELATED_DISPUTE edge: {empty_with_edges}")
    print("=> Nay la van de! Edge duoc tao ra nhung law khong ghi ro loai tranh chap.")


def check_case_description_quality(data):
    """Check the quality of case descriptions"""
    nodes_data = data["nodes_data"]

    print("\n" + "=" * 70)
    print("3. KIEM TRA CHAT LUONG MO TA CASE NODES")
    print("=" * 70)

    cases = {nid: ninfo for nid, ninfo in nodes_data.items() if ninfo["type"] == "Cases"}

    # Check for generic/template descriptions
    generic_patterns = [
        "Parties Info: Cá nhân, Tổ chức. Dispute Acts: Tranh chấp quyền lợi, Vi phạm quy định pháp luật",
        "Parties Info: Người lao động, Người sử dụng lao động. Dispute Acts: Tranh chấp quyền lợi",
        "Parties Info: Người tham gia BHXH, Cơ quan BHXH. Dispute Acts: Tranh chấp quyền lợi",
    ]

    generic_count = Counter()
    specific_count = 0

    for cid, cinfo in cases.items():
        desc = cinfo["data"].get("description", "")
        is_generic = False
        for pattern in generic_patterns:
            if pattern in str(desc):
                generic_count[pattern[:60]] += 1
                is_generic = True
                break
        if not is_generic:
            specific_count += 1

    total = len(cases)
    total_generic = sum(generic_count.values())
    print(f"\nTong cases: {total}")
    print(
        f"Cases co mo ta CU THE (co tinh huong thuc): {specific_count} ({100*specific_count/total:.1f}%)"
    )
    print(
        f"Cases co mo ta CHUNG CHUNG (template): {total_generic} ({100*total_generic/total:.1f}%)"
    )

    print("\nPhan loai template:")
    for pattern, count in generic_count.most_common():
        print(f"  '{pattern}...': {count} cases")

    # Show some specific cases
    print("\n  3 mau Case cu the (co noi dung thuc):")
    count = 0
    for cid, cinfo in cases.items():
        desc = str(cinfo["data"].get("description", ""))
        is_generic = any(p in desc for p in generic_patterns)
        if not is_generic and len(desc) > 50:
            print(f"    -> {desc[:200]}")
            count += 1
            if count >= 3:
                break


def check_cluster_balance(data):
    """Check cluster size distribution"""
    graph = data["graph"]
    nodes_data = data["nodes_data"]

    print("\n" + "=" * 70)
    print("4. KIEM TRA PHAN PHOI CLUSTER")
    print("=" * 70)

    clusters = defaultdict(list)
    for u, v, d in graph.edges(data=True):
        if d.get("relation_type") == "BELONGS_TO_CLUSTER":
            clusters[v].append(u)

    print(f"\nPhan phoi kich thuoc cluster:")
    sizes = []
    for cid, cases in sorted(clusters.items()):
        cdata = nodes_data.get(cid, {}).get("data", {})
        summary = str(cdata.get("summary", ""))[:100]
        print(f"  Cluster {cid[:8]}...: {len(cases):>5} cases  |  {summary}")
        sizes.append(len(cases))

    print(f"\n  Min: {min(sizes)}, Max: {max(sizes)}, Mean: {np.mean(sizes):.0f}")
    print(f"  Top 1 cluster chiem: {max(sizes)/sum(sizes)*100:.1f}% tong so case")

    # Check: is Cluster 7 (3412 cases) too dominant?
    if max(sizes) / sum(sizes) > 0.5:
        print("  !! CANH BAO: 1 cluster chiem qua lon => Louvain co the khong tach tot")


def check_case_without_law(data):
    """Check cases that have no RELATES_TO_LAW edges"""
    graph = data["graph"]
    nodes_data = data["nodes_data"]

    print("\n" + "=" * 70)
    print("5. KIEM TRA CASE KHONG CO QUAN HE VOI LAW")
    print("=" * 70)

    cases = {nid for nid, ninfo in nodes_data.items() if ninfo["type"] == "Cases"}
    cases_with_law = set()
    for u, v, d in graph.edges(data=True):
        if d.get("relation_type") == "RELATES_TO_LAW":
            cases_with_law.add(u)

    cases_without_law = cases - cases_with_law
    print(f"\nTong cases: {len(cases)}")
    print(
        f"Cases CO quan he voi law: {len(cases_with_law)} ({100*len(cases_with_law)/len(cases):.1f}%)"
    )
    print(
        f"Cases KHONG co quan he voi law: {len(cases_without_law)} ({100*len(cases_without_law)/len(cases):.1f}%)"
    )

    # Show some examples
    if cases_without_law:
        print("\n  3 mau Case KHONG co law:")
        for cid in list(cases_without_law)[:3]:
            cdata = nodes_data[cid]["data"]
            desc = str(cdata.get("description", "N/A"))[:200]
            laws_field = cdata.get("law", cdata.get("laws", []))
            print(f"    ID: {cid[:30]}...")
            print(f"    Desc: {desc}")
            print(f"    Laws field in data: {str(laws_field)[:100]}")
            print()


def check_law_entry_quality(data):
    """Check law entry names for quality"""
    nodes_data = data["nodes_data"]

    print("\n" + "=" * 70)
    print("6. KIEM TRA CHAT LUONG MA DIEU LUAT")
    print("=" * 70)

    laws = {nid: ninfo for nid, ninfo in nodes_data.items() if ninfo["type"] == "Laws"}

    # Categorize law entry formats
    pure_number = 0
    zalo_format = 0
    dieu_format = 0
    other_format = 0

    for lid, linfo in laws.items():
        entry = str(linfo["data"].get("entry", lid))
        if entry.isdigit():
            pure_number += 1
        elif entry.startswith("zalo_"):
            zalo_format += 1
        elif "Điều" in entry or "điều" in entry:
            dieu_format += 1
        else:
            other_format += 1

    print(f"\nTong laws: {len(laws)}")
    print(f"  So dang (VD: '584'):          {pure_number} ({100*pure_number/len(laws):.1f}%)")
    print(f"  Zalo format (VD: 'zalo_...'):  {zalo_format} ({100*zalo_format/len(laws):.1f}%)")
    print(f"  Dieu format (VD: 'Điều 584'): {dieu_format} ({100*dieu_format/len(laws):.1f}%)")
    print(f"  Khac:                          {other_format}")

    # Show some zalo format samples
    if zalo_format > 0:
        print("\n  5 mau law format 'zalo_...':")
        count = 0
        for lid, linfo in laws.items():
            entry = str(linfo["data"].get("entry", lid))
            if entry.startswith("zalo_"):
                desc = str(linfo["data"].get("description", ""))[:100]
                print(f"    Entry: {entry}")
                print(f"    Desc: {desc}")
                count += 1
                if count >= 5:
                    break


def main():
    random.seed(42)
    print("Loading graph...")
    data = load_graph(GRAPH_PATH)
    print("Done.\n")

    check_dispute_names(data)
    check_law_dispute_field_empty(data)
    check_case_description_quality(data)
    check_cluster_balance(data)
    check_case_without_law(data)
    check_law_entry_quality(data)

    print("\n" + "=" * 70)
    print("KIEM TRA HOAN TAT")
    print("=" * 70)


if __name__ == "__main__":
    main()
