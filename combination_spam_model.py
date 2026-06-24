from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import joblib

from transformer_spam_model import predict_transformer_spam


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


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


@lru_cache(maxsize=1)
def _load_config(config_path_text: str) -> dict:
    config_path = Path(config_path_text)
    if not config_path.exists():
        raise FileNotFoundError(f"Combination model config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=2)
def _load_joblib_model(model_path_text: str):
    model_path = Path(model_path_text)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def _resolve_model_path(config_path: Path, relative_path: str) -> Path:
    project_root = config_path.resolve().parents[2]
    return project_root / relative_path


def _normalise_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    return {
        "ham": float(probabilities.get("ham", 0.0)),
        "spam": float(probabilities.get("spam", 0.0)),
    }


def _component_from_probabilities(probabilities: dict[str, float]) -> ComponentPrediction:
    probabilities = _normalise_probabilities(probabilities)
    label = "spam" if probabilities["spam"] >= probabilities["ham"] else "ham"
    return ComponentPrediction(
        label=label,
        confidence=max(probabilities.values()),
        probabilities=probabilities,
        spam_probability=probabilities["spam"],
    )


def _predict_classical_component(model, message: str) -> ComponentPrediction:
    classes = [str(label).lower() for label in model.classes_]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([message])[0]
        probability_by_label = {
            label: float(probability)
            for label, probability in zip(classes, probabilities)
        }
        return _component_from_probabilities(probability_by_label)

    if not hasattr(model, "decision_function"):
        prediction = str(model.predict([message])[0]).lower()
        probabilities = {
            "ham": 1.0 if prediction == "ham" else 0.0,
            "spam": 1.0 if prediction == "spam" else 0.0,
        }
        return _component_from_probabilities(probabilities)

    score = float(model.decision_function([message])[0])
    spam_probability = _sigmoid(score)
    if len(classes) >= 2 and classes[1] != "spam":
        spam_probability = 1.0 - spam_probability

    probabilities = {
        "ham": 1.0 - spam_probability,
        "spam": spam_probability,
    }
    return _component_from_probabilities(probabilities)


def predict_combination_spam(config_path, message: str) -> CombinationPrediction:
    config_path = Path(config_path)
    config = _load_config(str(config_path.resolve()))

    model_paths = config["models"]
    svm_model = _load_joblib_model(str(_resolve_model_path(config_path, model_paths["svm"])))
    logistic_regression_model = _load_joblib_model(
        str(_resolve_model_path(config_path, model_paths["logistic_regression"]))
    )

    svm_prediction = _predict_classical_component(svm_model, message)
    logistic_regression_prediction = _predict_classical_component(logistic_regression_model, message)
    transformer_prediction = predict_transformer_spam(
        _resolve_model_path(config_path, model_paths["xlm_roberta"]),
        message,
        max_length=int(config.get("max_length", 160)),
    )
    xlm_prediction = _component_from_probabilities(transformer_prediction.probabilities)

    component_predictions = {
        "svm": svm_prediction,
        "logistic_regression": logistic_regression_prediction,
        "xlm_roberta": xlm_prediction,
    }
    weights = {key: float(value) for key, value in config["weights"].items()}
    threshold = float(config["threshold"])

    spam_probability = sum(
        weights[key] * component_predictions[key].spam_probability
        for key in weights
    )
    spam_probability = max(0.0, min(spam_probability, 1.0))
    probabilities = {
        "ham": 1.0 - spam_probability,
        "spam": spam_probability,
    }
    label = "spam" if spam_probability >= threshold else "ham"
    confidence = spam_probability if label == "spam" else 1.0 - spam_probability

    return CombinationPrediction(
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        component_predictions=component_predictions,
        weights=weights,
        threshold=threshold,
    )
