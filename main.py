from spam_project.model_catalog import (
    COMBINATION_COMPONENT_LABELS,
    COMBINATION_MODEL_NAME,
    MODEL_SPECS,
    MULTILINGUAL_TRANSFORMER_MODEL_NAME,
    TRANSFORMER_MODEL_NAME,
)
from spam_project.models.registry import ModelRegistry
from spam_project.ui.app import SpamDetectorApp, create_app, run_app


_MODEL_REGISTRY = ModelRegistry()
MODEL_OPTIONS = {spec.display_name: spec.path for spec in MODEL_SPECS}
MODEL_METRICS = _MODEL_REGISTRY.list_metrics_by_display_name()

__all__ = [
    "COMBINATION_COMPONENT_LABELS",
    "COMBINATION_MODEL_NAME",
    "MODEL_METRICS",
    "MODEL_OPTIONS",
    "MULTILINGUAL_TRANSFORMER_MODEL_NAME",
    "SpamDetectorApp",
    "TRANSFORMER_MODEL_NAME",
    "create_app",
    "run_app",
]


if __name__ == "__main__":
    run_app()
