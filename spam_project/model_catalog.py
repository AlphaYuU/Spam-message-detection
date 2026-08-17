from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(sys.executable).resolve().parent
    if getattr(sys, "frozen", False)
    else Path(__file__).resolve().parents[1]
)
MODELS_DIR = PROJECT_ROOT / "Models"
COMBINATION_CONFIG_PATH = MODELS_DIR / "combination_model" / "config.json"

TRANSFORMER_MODEL_NAME = "XLM-RoBERTa"
MULTILINGUAL_TRANSFORMER_MODEL_NAME = "Multi-language"
COMBINATION_MODEL_NAME = "Combination Model"
SVM_MODEL_NAME = "Support Vector Machine"
LOGISTIC_REGRESSION_MODEL_NAME = "Logistic Regression"
DEFAULT_MODEL_DISPLAY_NAME = MULTILINGUAL_TRANSFORMER_MODEL_NAME

DEFAULT_TRANSFORMER_MAX_LENGTH = 128
MULTILINGUAL_TRANSFORMER_MAX_LENGTH = 192
COMBINATION_TRANSFORMER_MAX_LENGTH = 160

REQUIRED_COMBINATION_COMPONENTS = (
    "svm",
    "logistic_regression",
    "xlm_roberta",
)
COMBINATION_COMPONENT_LABELS = {
    "svm": "SVM",
    "logistic_regression": "LR",
    "xlm_roberta": "XLM",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    kind: str
    path: Path
    fallback_metrics: dict[str, float] = field(default_factory=dict)
    max_length: int | None = None
    explainable: bool = False
    metrics_path: Path | None = None
    combination_metrics_key: str | None = None

    @property
    def metrics(self) -> dict[str, float]:
        """Compatibility fallback metrics; ModelRegistry loads preferred metric sources."""
        return dict(self.fallback_metrics)


MODEL_SPECS = (
    ModelSpec(
        model_id="xlm_roberta",
        display_name=TRANSFORMER_MODEL_NAME,
        kind="transformer",
        path=MODELS_DIR / "transformer_spam_classifier",
        fallback_metrics={
            "accuracy": 0.988361,
            "precision": 0.975410,
            "recall": 0.929688,
        },
        max_length=DEFAULT_TRANSFORMER_MAX_LENGTH,
        combination_metrics_key="xlm_roberta",
    ),
    ModelSpec(
        model_id="multi_language",
        display_name=MULTILINGUAL_TRANSFORMER_MODEL_NAME,
        kind="transformer",
        path=MODELS_DIR / "multilingual_transformer_spam_classifier",
        fallback_metrics={
            "accuracy": 0.993324,
            "precision": 0.963087,
            "recall": 0.972881,
        },
        max_length=MULTILINGUAL_TRANSFORMER_MAX_LENGTH,
        metrics_path=MODELS_DIR / "multilingual_transformer_spam_classifier" / "metrics.json",
    ),
    ModelSpec(
        model_id="combination_model",
        display_name=COMBINATION_MODEL_NAME,
        kind="combination",
        path=COMBINATION_CONFIG_PATH,
        combination_metrics_key="combination_model",
    ),
    ModelSpec(
        model_id="svm",
        display_name=SVM_MODEL_NAME,
        kind="classical",
        path=MODELS_DIR / "svm_spam_classifier.joblib",
        fallback_metrics={
            "accuracy": 0.984481,
            "precision": 0.982759,
            "recall": 0.890625,
        },
        explainable=True,
        combination_metrics_key="svm",
    ),
    ModelSpec(
        model_id="logistic_regression",
        display_name=LOGISTIC_REGRESSION_MODEL_NAME,
        kind="classical",
        path=MODELS_DIR / "logistic_regression_spam_classifier.joblib",
        fallback_metrics={
            "accuracy": 0.965082,
            "precision": 0.979167,
            "recall": 0.734375,
        },
        explainable=True,
        combination_metrics_key="logistic_regression",
    ),
)

MODEL_SPECS_BY_ID = {spec.model_id: spec for spec in MODEL_SPECS}
MODEL_IDS_BY_DISPLAY_NAME = {spec.display_name: spec.model_id for spec in MODEL_SPECS}
