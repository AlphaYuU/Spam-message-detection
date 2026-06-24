from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExplanationItem:
    feature: str
    contribution: float
    tendency: str
    tfidf_score: float | None = None
    model_weight: float | None = None
    method: str = "unknown"


@dataclass(frozen=True)
class ComponentScore:
    label: str
    confidence: float
    probabilities: dict[str, float]
    spam_probability: float
    weight: float | None = None
    display_name: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]
    explanations: list[ExplanationItem] = field(default_factory=list)
    components: dict[str, ComponentScore] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

