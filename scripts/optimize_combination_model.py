from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "Dataset source" / "training" / "Dataset_SMS_clean.csv"
MODELS_DIR = BASE_DIR / "Models"
CONFIG_DIR = MODELS_DIR / "combination_model"
CONFIG_PATH = CONFIG_DIR / "config.json"

SVM_MODEL_PATH = MODELS_DIR / "svm_spam_classifier.joblib"
LOGISTIC_REGRESSION_MODEL_PATH = MODELS_DIR / "logistic_regression_spam_classifier.joblib"
TRANSFORMER_MODEL_PATH = MODELS_DIR / "transformer_spam_classifier"

TEST_SIZE = 0.2
VALIDATION_SIZE_FROM_HOLDOUT = 0.5
RANDOM_STATE = 42
MAX_LENGTH = 160
MIN_XLM_ROBERTA_WEIGHT = 0.45
MAX_XLM_ROBERTA_WEIGHT = 0.60
MIN_CLASSICAL_MODEL_WEIGHT = 0.10
LABEL2ID = {"ham": 0, "spam": 1}
ID2LABEL = {0: "ham", 1: "spam"}


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    positive = values >= 0
    result = np.empty_like(values, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def load_data() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    df = df.dropna(subset=["Category", "Message"]).copy()
    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
    df["Message"] = df["Message"].astype(str).str.strip()
    df = df[df["Category"].isin(LABEL2ID)]
    df = df[df["Message"] != ""]
    df["label"] = df["Category"].map(LABEL2ID)

    _, x_holdout, _, y_holdout = train_test_split(
        df["Message"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=VALIDATION_SIZE_FROM_HOLDOUT,
        random_state=RANDOM_STATE,
        stratify=y_holdout,
    )
    return (
        x_validation.reset_index(drop=True),
        x_test.reset_index(drop=True),
        y_validation.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def get_classical_spam_probability(model, messages: pd.Series) -> np.ndarray:
    messages_list = messages.tolist()
    classes = [str(label).lower() for label in model.classes_]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(messages_list)
        spam_index = classes.index("spam")
        return probabilities[:, spam_index].astype(float)

    if not hasattr(model, "decision_function"):
        raise TypeError("Model must expose predict_proba or decision_function.")

    scores = np.asarray(model.decision_function(messages_list), dtype=float)
    if scores.ndim > 1:
        spam_index = classes.index("spam")
        return sigmoid(scores[:, spam_index])

    spam_probability = sigmoid(scores)
    if len(classes) >= 2 and classes[1] != "spam":
        spam_probability = 1.0 - spam_probability
    return spam_probability.astype(float)


@torch.inference_mode()
def get_transformer_spam_probability(messages: pd.Series) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_PATH, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    id_to_label = {int(index): str(label).lower() for index, label in model.config.id2label.items()}
    spam_index = next(index for index, label in id_to_label.items() if label == "spam")

    spam_probabilities = []
    batch_size = 32
    messages_list = messages.tolist()
    for start in range(0, len(messages_list), batch_size):
        batch = messages_list[start : start + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)
        spam_probabilities.extend(probabilities[:, spam_index].detach().cpu().numpy().tolist())
    return np.asarray(spam_probabilities, dtype=float)


def probability_to_predictions(spam_probability: np.ndarray, threshold: float) -> np.ndarray:
    return (spam_probability >= threshold).astype(int)


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=LABEL2ID["spam"],
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "spam_precision": float(precision),
        "spam_recall": float(recall),
        "spam_f1": float(f1),
    }


def score_probability(y_true: np.ndarray, spam_probability: np.ndarray, threshold: float) -> dict[str, float]:
    return score_predictions(y_true, probability_to_predictions(spam_probability, threshold))


def search_weights(
    y_validation: np.ndarray,
    svm_probability: np.ndarray,
    logistic_regression_probability: np.ndarray,
    transformer_probability: np.ndarray,
) -> dict:
    best = None
    grid = np.arange(0.0, 1.0001, 0.05)
    thresholds = np.arange(0.05, 0.9501, 0.025)

    for svm_weight in grid:
        for logistic_regression_weight in grid:
            transformer_weight = 1.0 - svm_weight - logistic_regression_weight
            if transformer_weight < -1e-9:
                continue
            if transformer_weight + 1e-9 < MIN_XLM_ROBERTA_WEIGHT:
                continue
            if transformer_weight - 1e-9 > MAX_XLM_ROBERTA_WEIGHT:
                continue
            if svm_weight + 1e-9 < MIN_CLASSICAL_MODEL_WEIGHT:
                continue
            if logistic_regression_weight + 1e-9 < MIN_CLASSICAL_MODEL_WEIGHT:
                continue
            if transformer_weight + 1e-9 < svm_weight or transformer_weight + 1e-9 < logistic_regression_weight:
                continue

            weights = np.array([svm_weight, logistic_regression_weight, transformer_weight], dtype=float)
            combined_probability = (
                weights[0] * svm_probability
                + weights[1] * logistic_regression_probability
                + weights[2] * transformer_probability
            )

            for threshold in thresholds:
                metrics = score_probability(y_validation, combined_probability, float(threshold))
                candidate = {
                    "weights": {
                        "svm": float(weights[0]),
                        "logistic_regression": float(weights[1]),
                        "xlm_roberta": float(weights[2]),
                    },
                    "threshold": float(threshold),
                    "metrics": metrics,
                }
                key = (
                    metrics["macro_f1"],
                    metrics["spam_f1"],
                    metrics["accuracy"],
                    weights[2],
                )
                if best is None or key > best["key"]:
                    candidate["key"] = key
                    best = candidate

    best.pop("key", None)
    return best


def main() -> None:
    x_validation, x_test, y_validation, y_test = load_data()
    y_validation_array = y_validation.to_numpy()
    y_test_array = y_test.to_numpy()

    svm_model = joblib.load(SVM_MODEL_PATH)
    logistic_regression_model = joblib.load(LOGISTIC_REGRESSION_MODEL_PATH)

    print("Scoring validation split...")
    validation_probabilities = {
        "svm": get_classical_spam_probability(svm_model, x_validation),
        "logistic_regression": get_classical_spam_probability(logistic_regression_model, x_validation),
        "xlm_roberta": get_transformer_spam_probability(x_validation),
    }

    print("Searching weights...")
    best = search_weights(
        y_validation_array,
        validation_probabilities["svm"],
        validation_probabilities["logistic_regression"],
        validation_probabilities["xlm_roberta"],
    )

    print("Scoring test split...")
    test_probabilities = {
        "svm": get_classical_spam_probability(svm_model, x_test),
        "logistic_regression": get_classical_spam_probability(logistic_regression_model, x_test),
        "xlm_roberta": get_transformer_spam_probability(x_test),
    }
    weights = best["weights"]
    threshold = best["threshold"]
    combined_test_probability = (
        weights["svm"] * test_probabilities["svm"]
        + weights["logistic_regression"] * test_probabilities["logistic_regression"]
        + weights["xlm_roberta"] * test_probabilities["xlm_roberta"]
    )

    model_metrics = {
        "svm": {
            "validation": score_probability(y_validation_array, validation_probabilities["svm"], 0.5),
            "test": score_probability(y_test_array, test_probabilities["svm"], 0.5),
        },
        "logistic_regression": {
            "validation": score_probability(y_validation_array, validation_probabilities["logistic_regression"], 0.5),
            "test": score_probability(y_test_array, test_probabilities["logistic_regression"], 0.5),
        },
        "xlm_roberta": {
            "validation": score_probability(y_validation_array, validation_probabilities["xlm_roberta"], 0.5),
            "test": score_probability(y_test_array, test_probabilities["xlm_roberta"], 0.5),
        },
        "combination_model": {
            "validation": best["metrics"],
            "test": score_probability(y_test_array, combined_test_probability, threshold),
        },
    }

    config = {
        "name": "Combination Model",
        "description": "Weighted soft-voting ensemble over SVM, Logistic Regression, and English XLM-RoBERTa.",
        "weights": weights,
        "threshold": threshold,
        "objective": "validation_macro_f1_with_balanced_xlm_primary_constraint",
        "constraints": {
            "min_xlm_roberta_weight": MIN_XLM_ROBERTA_WEIGHT,
            "max_xlm_roberta_weight": MAX_XLM_ROBERTA_WEIGHT,
            "min_classical_model_weight": MIN_CLASSICAL_MODEL_WEIGHT,
            "xlm_roberta_must_be_largest_weight": True,
        },
        "max_length": MAX_LENGTH,
        "label2id": LABEL2ID,
        "id2label": {str(key): value for key, value in ID2LABEL.items()},
        "models": {
            "svm": str(SVM_MODEL_PATH.relative_to(BASE_DIR)),
            "logistic_regression": str(LOGISTIC_REGRESSION_MODEL_PATH.relative_to(BASE_DIR)),
            "xlm_roberta": str(TRANSFORMER_MODEL_PATH.relative_to(BASE_DIR)),
        },
        "splits": {
            "source_dataset": str(DATASET_PATH.relative_to(BASE_DIR)),
            "holdout_size": TEST_SIZE,
            "validation_size_from_holdout": VALIDATION_SIZE_FROM_HOLDOUT,
            "random_state": RANDOM_STATE,
            "validation_rows": int(len(x_validation)),
            "test_rows": int(len(x_test)),
            "validation_spam_rows": int(y_validation.sum()),
            "test_spam_rows": int(y_test.sum()),
        },
        "metrics": model_metrics,
    }

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(config["weights"], indent=2))
    print("threshold:", config["threshold"])
    print("validation:", json.dumps(model_metrics["combination_model"]["validation"], indent=2))
    print("test:", json.dumps(model_metrics["combination_model"]["test"], indent=2))
    print("Saved:", CONFIG_PATH)


if __name__ == "__main__":
    main()
