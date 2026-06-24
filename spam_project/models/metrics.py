from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

from spam_project.model_catalog import COMBINATION_CONFIG_PATH, ModelSpec


@lru_cache(maxsize=16)
def _load_json(path_text: str) -> dict:
    path = Path(path_text)
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_display_metrics(spec: ModelSpec) -> dict[str, float]:
    if spec.metrics_path and spec.metrics_path.exists():
        metrics = _extract_metrics_from_metrics_json(spec.metrics_path)
        if metrics:
            return metrics

    if spec.combination_metrics_key and COMBINATION_CONFIG_PATH.exists():
        metrics = _extract_metrics_from_combination_config(spec.combination_metrics_key)
        if metrics:
            return metrics

    return dict(spec.fallback_metrics)


def _extract_metrics_from_metrics_json(metrics_path: Path) -> dict[str, float]:
    try:
        payload = _load_json(str(metrics_path.resolve()))
    except (OSError, json.JSONDecodeError):
        return {}

    metrics = payload.get("test_metrics", payload)
    return _normalise_metric_keys(metrics, prefixes=("test_", ""))


def _extract_metrics_from_combination_config(model_key: str) -> dict[str, float]:
    try:
        payload = _load_json(str(COMBINATION_CONFIG_PATH.resolve()))
    except (OSError, json.JSONDecodeError):
        return {}

    model_metrics = payload.get("metrics", {}).get(model_key, {})
    metrics = model_metrics.get("test", model_metrics)
    return _normalise_metric_keys(metrics, prefixes=("",))


def _normalise_metric_keys(metrics: dict, prefixes: tuple[str, ...]) -> dict[str, float]:
    normalised = {}
    key_candidates = {
        "accuracy": ("accuracy",),
        "precision": ("precision", "spam_precision"),
        "recall": ("recall", "spam_recall"),
    }
    for output_key, base_candidates in key_candidates.items():
        value = _first_metric_value(metrics, base_candidates, prefixes)
        if value is not None:
            normalised[output_key] = float(value)
    return normalised


def _first_metric_value(metrics: dict, base_candidates: tuple[str, ...], prefixes: tuple[str, ...]):
    for prefix in prefixes:
        for base_key in base_candidates:
            key = f"{prefix}{base_key}"
            if key in metrics:
                return metrics[key]
    return None

