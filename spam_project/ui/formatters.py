from __future__ import annotations


def format_probability_status(probabilities):
    return " | ".join(
        f"{label.upper()}: {probability:.2%}"
        for label, probability in sorted((probabilities or {}).items())
    )


def format_combination_status(result):
    probability_text = format_probability_status(result.probabilities)
    weights = result.metadata.get("weights", {})
    if not weights:
        return probability_text

    weight_text = (
        f"Weights - SVM: {weights.get('svm', 0.0):.2f}, "
        f"LR: {weights.get('logistic_regression', 0.0):.2f}, "
        f"XLM: {weights.get('xlm_roberta', 0.0):.2f}"
    )
    threshold = result.metadata.get("threshold")
    threshold_text = ""
    if threshold is not None:
        threshold_text = f" | Threshold: {float(threshold):.3f}"
    return f"{probability_text} | {weight_text}{threshold_text}"

