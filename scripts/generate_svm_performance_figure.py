from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "Dataset source" / "training" / "Dataset_SMS_clean.csv"
MODEL_PATH = BASE_DIR / "Models" / "svm_spam_classifier.joblib"
CONFIG_PATH = BASE_DIR / "Models" / "combination_model" / "config.json"
OUTPUT_PATH = BASE_DIR / "figure" / "figure_4_1_svm_performance.png"

TEST_SIZE = 0.2
VALIDATION_SIZE_FROM_HOLDOUT = 0.5
RANDOM_STATE = 42


def load_splits() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Reproduce the English split used by the combination-model evaluation."""
    frame = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    frame = frame.dropna(subset=["Category", "Message"]).copy()
    frame["Category"] = frame["Category"].astype(str).str.strip().str.lower()
    frame["Message"] = frame["Message"].astype(str).str.strip()
    frame = frame[frame["Category"].isin(["ham", "spam"])]
    frame = frame[frame["Message"] != ""]

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        frame["Message"],
        frame["Category"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=frame["Category"],
    )
    _, x_test, _, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=VALIDATION_SIZE_FROM_HOLDOUT,
        random_state=RANDOM_STATE,
        stratify=y_holdout,
    )
    return x_train, x_test, y_train, y_test


def verify_against_saved_metrics(accuracy: float, report: dict) -> None:
    saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["metrics"]["svm"]["test"]
    observed = {
        "accuracy": accuracy,
        "spam_precision": report["spam"]["precision"],
        "spam_recall": report["spam"]["recall"],
        "spam_f1": report["spam"]["f1-score"],
    }
    for key, value in observed.items():
        if not np.isclose(value, saved[key], atol=1e-12):
            raise ValueError(f"Computed {key}={value} does not match saved value {saved[key]}")


def main() -> None:
    x_train, x_test, y_train, y_test = load_splits()
    model = joblib.load(MODEL_PATH)

    train_predictions = model.predict(x_train.tolist())
    test_predictions = model.predict(x_test.tolist())
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    report = classification_report(
        y_test,
        test_predictions,
        labels=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )
    verify_against_saved_metrics(test_accuracy, report)

    sns.set_theme(style="whitegrid")

    accuracy_df = pd.DataFrame(
        {
            "Dataset": ["Train", "Test"],
            "Accuracy": [train_accuracy, test_accuracy],
        }
    )
    score_ticks = np.arange(0.8, 1.0001, 0.025)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(
        data=accuracy_df,
        x="Dataset",
        y="Accuracy",
        hue="Dataset",
        palette=["blue", "red"],
        ax=axes[0],
        legend=False,
    )
    axes[0].set_ylim(0.8, 1.01)
    axes[0].set_yticks(score_ticks)
    axes[0].set_title("Training vs Test Accuracy")
    axes[0].set_ylabel("Accuracy")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.4f", padding=2)

    metrics_df = pd.DataFrame(report).transpose().loc[
        ["ham", "spam"], ["precision", "recall", "f1-score"]
    ].transpose()
    metrics_df.plot(kind="bar", ax=axes[1], color=["blue", "red"])
    axes[1].set_ylim(0.8, 1.01)
    axes[1].set_yticks(score_ticks)
    axes[1].set_title("Precision, Recall, and F1-Score")
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(title="Class")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.4f", padding=2)

    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Training rows: {len(x_train)} | Final evaluation rows: {len(x_test)}")
    print(f"Training accuracy: {train_accuracy:.6f} | Final evaluation accuracy: {test_accuracy:.6f}")
    print(
        "Ham P/R/F1: "
        f"{report['ham']['precision']:.6f}/"
        f"{report['ham']['recall']:.6f}/"
        f"{report['ham']['f1-score']:.6f}"
    )
    print(
        "Spam P/R/F1: "
        f"{report['spam']['precision']:.6f}/"
        f"{report['spam']['recall']:.6f}/"
        f"{report['spam']['f1-score']:.6f}"
    )


if __name__ == "__main__":
    main()
