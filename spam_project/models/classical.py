from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import joblib

from spam_project.explainers.tfidf_linear import explain_linear_tfidf_model
from spam_project.models.types import ComponentScore, PredictionResult


@lru_cache(maxsize=4)
def load_joblib_model(model_path_text: str):
    model_path = Path(model_path_text)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_classical_result(model_path, message: str, display_name: str, explain: bool = True) -> PredictionResult:
    model = load_joblib_model(str(Path(model_path).resolve()))
    prediction = str(model.predict([message])[0]).lower()
    confidence, probabilities = get_classical_confidence(model, message)
    explanations = []
    metadata = {
        "display_name": display_name,
        "model_path": str(Path(model_path)),
    }

    if explain:
        try:
            explanations = explain_linear_tfidf_model(model, message)
        except (TypeError, ValueError) as error:
            metadata["explanation_error"] = str(error)

    return PredictionResult(
        label=prediction,
        confidence=float(confidence if confidence is not None else 1.0),
        probabilities=probabilities or {},
        explanations=explanations,
        metadata=metadata,
    )


def predict_classical_component(model, message: str, display_name: str | None = None) -> ComponentScore:
    _, probabilities = get_classical_confidence(model, message)
    if not probabilities:
        prediction = str(model.predict([message])[0]).lower()
        probabilities = {
            "ham": 1.0 if prediction == "ham" else 0.0,
            "spam": 1.0 if prediction == "spam" else 0.0,
        }
    return component_from_probabilities(probabilities, display_name=display_name)


def component_from_probabilities(
    probabilities: dict[str, float],
    display_name: str | None = None,
    weight: float | None = None,
    metadata: dict | None = None,
) -> ComponentScore:
    probabilities = normalise_probabilities(probabilities)
    label = "spam" if probabilities["spam"] >= probabilities["ham"] else "ham"
    return ComponentScore(
        label=label,
        confidence=max(probabilities.values()),
        probabilities=probabilities,
        spam_probability=probabilities["spam"],
        weight=weight,
        display_name=display_name,
        metadata=metadata or {},
    )


def normalise_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    return {
        "ham": float(probabilities.get("ham", 0.0)),
        "spam": float(probabilities.get("spam", 0.0)),
    }


def get_classical_confidence(model, message: str):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([message])[0]
        classes = [str(label).lower() for label in model.classes_]
        probability_by_label = {
            label: float(probability)
            for label, probability in zip(classes, probabilities)
        }
        return max(probability_by_label.values()), probability_by_label

    if hasattr(model, "decision_function"):
        score = float(model.decision_function([message])[0])
        confidence = _sigmoid(abs(score))
        classes = [str(label).lower() for label in model.classes_]
        if len(classes) == 2:
            positive_label = classes[1]
            negative_label = classes[0]
            if score >= 0:
                probabilities = {
                    positive_label: confidence,
                    negative_label: 1.0 - confidence,
                }
            else:
                probabilities = {
                    negative_label: confidence,
                    positive_label: 1.0 - confidence,
                }
            return confidence, probabilities

    return None, None


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)

