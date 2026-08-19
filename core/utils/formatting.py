from typing import Dict, Any


def concat_feature_descriptions(description: Dict[str, Any]) -> str:
    """
    Concatenate case features into a single descriptive string for retrieval queries.
    """
    res = ""
    res += "Parties Info: " + ", ".join(description.get("parties_info", [])) + ". "
    res += "Dispute Acts: " + ", ".join(description.get("dispute_acts", [])) + ". "
    res += "Subject Matter: " + ", ".join(description.get("subject_matter", [])) + ". "
    res += "Fault and Evidence: " + ", ".join(description.get("fault_and_evidence", [])) + ". "
    return res
