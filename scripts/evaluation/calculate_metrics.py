import json
import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from core.utils.logger import logger


def calculate_metrics(results_file):
    if not os.path.exists(results_file):
        logger.error(f"Results file not found: {results_file}")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        logger.warning("Results file is empty.")
        return

    y_true_charge = []
    y_pred_charge = []

    # We will simply collect all predicted charges for all defendants in a case
    # If ANY of the predicted charges match ANY of the true charges, we count it as correct
    # This is a simplified evaluation for demonstration. A more strict evaluation might
    # compare defendant-by-defendant.

    for case in results:
        true_disputes = case.get("true_dispute", [])
        if isinstance(true_disputes, str):
            true_disputes = [true_disputes]

        judge_res_list = case.get("judge_res", [])

        predicted_disputes = []
        for res in judge_res_list:
            # Look for the predicted dispute inside judge_result
            judge_result = res.get("judge_result", {})
            if isinstance(judge_result, dict):
                charges = judge_result.get("dispute_type", [])
                if isinstance(charges, list):
                    predicted_disputes.extend(charges)
                elif isinstance(charges, str):
                    predicted_disputes.append(charges)
            elif isinstance(judge_result, list):
                predicted_disputes.extend(judge_result)

        # Binary evaluation: did we predict at least one correct charge?
        # For a more rigorous evaluation, you'd want to flatten the lists and compare exact sets
        # But for this simple script, let's just see if there's an intersection

        if not true_disputes:
            continue

        # We need a unified format for sklearn. Let's just use the first true dispute and the first predicted
        # Or better, just calculate accuracy manually for multi-label.

        # Let's do a simple exact match (or at least one match) accuracy
        pred_set = set([str(p).lower().strip() for p in predicted_disputes])

        # If intersection exists, we consider it a 'match' for the dominant charge.
        # To use sklearn properly, we'd need to binarize all possible charges.

        # Let's just do a manual exact match accuracy for the primary charge for simplicity
        if true_disputes:
            primary_true = str(true_disputes[0]).lower().strip()
            y_true_charge.append(primary_true)

            # Find the best matching prediction or just take the first
            if primary_true in pred_set:
                y_pred_charge.append(primary_true)
            elif predicted_disputes:
                y_pred_charge.append(str(predicted_disputes[0]).lower().strip())
            else:
                y_pred_charge.append("no_prediction")

    if not y_true_charge:
        logger.warning("No valid cases to evaluate.")
        return

    acc = accuracy_score(y_true_charge, y_pred_charge)
    prec = precision_score(y_true_charge, y_pred_charge, average="macro", zero_division=0)
    rec = recall_score(y_true_charge, y_pred_charge, average="macro", zero_division=0)
    f1 = f1_score(y_true_charge, y_pred_charge, average="macro", zero_division=0)

    logger.info(f"--- Evaluation Metrics for {os.path.basename(results_file)} ---")
    logger.info(f"Total evaluated cases: {len(y_true_charge)}")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f} (Macro)")
    logger.info(f"Recall:    {rec:.4f} (Macro)")
    logger.info(f"F1 Score:  {f1:.4f} (Macro)")
    logger.info("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics from evaluate.py output")
    parser.add_argument("--file", type=str, required=True, help="Path to the JSON results file")
    args = parser.parse_args()

    calculate_metrics(args.file)
