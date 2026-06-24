from __future__ import annotations

from spam_project.model_catalog import MODEL_SPECS, MODEL_IDS_BY_DISPLAY_NAME, ModelSpec
from spam_project.models.classical import predict_classical_result
from spam_project.models.combination import (
    get_combination_weights,
    predict_combination_result,
    validate_combination_config,
)
from spam_project.models.metrics import load_display_metrics
from spam_project.models.transformer import predict_transformer_result


class ModelRegistry:
    def __init__(self, specs: tuple[ModelSpec, ...] = MODEL_SPECS):
        self._specs = tuple(specs)
        self._specs_by_id = {spec.model_id: spec for spec in self._specs}
        self._ids_by_display_name = {spec.display_name: spec.model_id for spec in self._specs}
        self._metrics_cache = {}

    def list_display_names(self) -> list[str]:
        return [spec.display_name for spec in self._specs]

    def get_spec(self, model_key: str) -> ModelSpec:
        model_id = self._resolve_model_id(model_key)
        return self._specs_by_id[model_id]

    def get_metrics(self, model_key: str) -> dict[str, float]:
        spec = self.get_spec(model_key)
        if spec.kind == "combination":
            return {}
        if spec.model_id not in self._metrics_cache:
            self._metrics_cache[spec.model_id] = load_display_metrics(spec)
        return dict(self._metrics_cache[spec.model_id])

    def list_metrics_by_display_name(self) -> dict[str, dict[str, float]]:
        metrics_by_name = {}
        for spec in self._specs:
            metrics = self.get_metrics(spec.model_id)
            if metrics:
                metrics_by_name[spec.display_name] = metrics
        return metrics_by_name

    def get_combination_weights(self, model_key: str = "combination_model") -> dict[str, float]:
        spec = self.get_spec(model_key)
        if spec.kind != "combination":
            return {}
        return get_combination_weights(spec.path)

    def predict(self, model_key: str, message: str):
        spec = self.get_spec(model_key)
        self.validate_model(model_key)

        if spec.kind == "combination":
            return predict_combination_result(spec.path, message)

        if spec.kind == "transformer":
            return predict_transformer_result(
                spec.path,
                message,
                display_name=spec.display_name,
                max_length=spec.max_length or 128,
            )

        if spec.kind == "classical":
            return predict_classical_result(
                spec.path,
                message,
                display_name=spec.display_name,
                explain=spec.explainable,
            )

        raise ValueError(f"Unsupported model kind: {spec.kind}")

    def validate_model(self, model_key: str) -> None:
        spec = self.get_spec(model_key)
        if not spec.path.exists():
            raise FileNotFoundError(f"Model file not found: {spec.path}")
        if spec.kind == "combination":
            validate_combination_config(spec.path)

    def _resolve_model_id(self, model_key: str) -> str:
        if model_key in self._specs_by_id:
            return model_key
        if model_key in self._ids_by_display_name:
            return self._ids_by_display_name[model_key]
        if model_key in MODEL_IDS_BY_DISPLAY_NAME:
            return MODEL_IDS_BY_DISPLAY_NAME[model_key]
        raise KeyError(f"Unknown model: {model_key}")
