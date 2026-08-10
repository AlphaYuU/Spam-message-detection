from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

from spam_project.explainers.tfidf_linear import explain_linear_tfidf_model
from spam_project.model_catalog import COMBINATION_COMPONENT_LABELS, REQUIRED_COMBINATION_COMPONENTS
from spam_project.models.classical import (
    component_from_probabilities,
    load_joblib_model,
    predict_classical_component,
)
from spam_project.models.transformer import predict_transformer_spam
from spam_project.models.types import ComponentScore, PredictionResult


@dataclass(frozen=True)
class ComponentPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    spam_probability: float


@dataclass(frozen=True)
class CombinationPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    component_predictions: dict[str, ComponentPrediction]
    weights: dict[str, float]
    threshold: float


@lru_cache(maxsize=4)
def load_combination_config(config_path_text: str) -> dict:
    config_path = Path(config_path_text)
    if not config_path.exists():
        raise FileNotFoundError(f"Combination model config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def predict_combination_spam(config_path, message: str) -> CombinationPrediction:
    result = predict_combination_result(config_path, message)
    component_predictions = {
        key: ComponentPrediction(
            label=component.label,
            confidence=component.confidence,
            probabilities=component.probabilities,
            spam_probability=component.spam_probability,
        )
        for key, component in result.components.items()
    }
    return CombinationPrediction(
        label=result.label,
        confidence=result.confidence,
        probabilities=result.probabilities,
        component_predictions=component_predictions,
        weights=dict(result.metadata.get("weights", {})),
        threshold=float(result.metadata.get("threshold", 0.5)),
    )


def predict_combination_result(config_path, message: str) -> PredictionResult:
    config_path = Path(config_path)
    config = load_combination_config(str(config_path.resolve()))
    _validate_combination_config_dict(config_path, config)
    weights = {
        key: float(config["weights"][key])
        for key in REQUIRED_COMBINATION_COMPONENTS
    }
    threshold = float(config["threshold"])
    component_scores = _predict_components(config_path, config, weights, message)

    spam_probability = sum(
        weights[key] * component_scores[key].spam_probability
        for key in weights
    )
    spam_probability = max(0.0, min(float(spam_probability), 1.0))
    probabilities = {
        "ham": 1.0 - spam_probability,
        "spam": spam_probability,
    }
    label = "spam" if spam_probability >= threshold else "ham"
    confidence = spam_probability if label == "spam" else 1.0 - spam_probability

    metadata = {
        "display_name": config.get("name", "Combination Model"),
        "config_path": str(config_path),
        "weights": weights,
        "threshold": threshold,
        "explanation_model": "svm",
        "explanation_method": "tfidf_linear",
    }
    explanations = []
    try:
        explanations = _explain_with_svm_component(config_path, config, message)
    except (TypeError, ValueError) as error:
        metadata["explanation_error"] = str(error)

    return PredictionResult(
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        explanations=explanations,
        components=component_scores,
        metadata=metadata,
    )


def get_combination_weights(config_path) -> dict[str, float]:
    config = load_combination_config(str(Path(config_path).resolve()))
    _validate_combination_config_dict(Path(config_path), config, validate_model_paths=False)
    return {
        component_key: float(config["weights"][component_key])
        for component_key in COMBINATION_COMPONENT_LABELS
    }


def validate_combination_config(config_path) -> None:
    config_path = Path(config_path)
    config = load_combination_config(str(config_path.resolve()))
    _validate_combination_config_dict(config_path, config)


def _predict_components(
    config_path: Path,
    config: dict,
    weights: dict[str, float],
    message: str,
) -> dict[str, ComponentScore]:
    model_paths = config["models"]
    svm_model = load_joblib_model(str(_resolve_model_path(config_path, model_paths["svm"])))
    logistic_regression_model = load_joblib_model(
        str(_resolve_model_path(config_path, model_paths["logistic_regression"]))
    )

    svm_prediction = predict_classical_component(
        svm_model,
        message,
        display_name=COMBINATION_COMPONENT_LABELS["svm"],
    )
    logistic_regression_prediction = predict_classical_component(
        logistic_regression_model,
        message,
        display_name=COMBINATION_COMPONENT_LABELS["logistic_regression"],
    )
    transformer_prediction = predict_transformer_spam(
        _resolve_model_path(config_path, model_paths["xlm_roberta"]),
        message,
        max_length=int(config.get("max_length", 160)),
    )
    xlm_prediction = component_from_probabilities(
        transformer_prediction.probabilities,
        display_name=COMBINATION_COMPONENT_LABELS["xlm_roberta"],
    )

    components = {
        "svm": svm_prediction,
        "logistic_regression": logistic_regression_prediction,
        "xlm_roberta": xlm_prediction,
    }
    return {
        key: ComponentScore(
            label=component.label,
            confidence=component.confidence,
            probabilities=component.probabilities,
            spam_probability=component.spam_probability,
            weight=weights.get(key),
            display_name=component.display_name,
            metadata=component.metadata,
        )
        for key, component in components.items()
    }


def _explain_with_svm_component(config_path: Path, config: dict, message: str):
    """Explain a combined prediction using its fitted SVM component."""
    svm_model_path = _resolve_model_path(config_path, config["models"]["svm"])
    svm_model = load_joblib_model(str(svm_model_path))
    return explain_linear_tfidf_model(svm_model, message)


def _resolve_model_path(config_path: Path, model_path_text: str) -> Path:
    model_path = Path(model_path_text)
    if model_path.is_absolute():
        return model_path
    project_root = config_path.resolve().parents[2]
    return project_root / model_path


def _validate_combination_config_dict(
    config_path: Path,
    config: dict,
    validate_model_paths: bool = True,
) -> None:
    for key in ("models", "weights", "threshold"):
        if key not in config:
            raise ValueError(f"Combination model config is missing '{key}': {config_path}")

    models = config.get("models", {})
    weights = config.get("weights", {})
    missing_models = [key for key in REQUIRED_COMBINATION_COMPONENTS if key not in models]
    missing_weights = [key for key in REQUIRED_COMBINATION_COMPONENTS if key not in weights]
    if missing_models:
        raise ValueError(f"Combination model config is missing model paths: {', '.join(missing_models)}")
    if missing_weights:
        raise ValueError(f"Combination model config is missing weights: {', '.join(missing_weights)}")

    for component_key in REQUIRED_COMBINATION_COMPONENTS:
        float(weights[component_key])
        if validate_model_paths:
            model_path = _resolve_model_path(config_path, models[component_key])
            if not model_path.exists():
                raise FileNotFoundError(f"Combination component model not found: {model_path}")
    float(config["threshold"])
