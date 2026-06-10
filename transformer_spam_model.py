from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys


DEFAULT_TRANSFORMER_PYTHON = Path("C:/ProgramData/anaconda3/envs/py310/python.exe")
PREDICTION_TIMEOUT_SECONDS = 300


@dataclass(frozen=True)
class TransformerPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class MissingTransformerDependencies(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_transformer_components(model_dir_text):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise MissingTransformerDependencies(
            "XLM-RoBERTa prediction requires torch and transformers. "
            "Run the UI from an environment that has both packages installed, such as the py310 conda environment."
        ) from error

    model_dir = Path(model_dir_text)
    if not model_dir.exists():
        raise FileNotFoundError(f"XLM-RoBERTa model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def _predict_transformer_spam_local(model_dir, message, max_length):
    torch, tokenizer, model, device = _load_transformer_components(str(Path(model_dir)))
    inputs = tokenizer(
        [message],
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        probabilities_tensor = torch.softmax(model(**inputs).logits, dim=-1)[0].detach().cpu()

    probabilities = probabilities_tensor.tolist()
    predicted_index = int(probabilities_tensor.argmax().item())
    id_to_label = {int(index): str(label).lower() for index, label in model.config.id2label.items()}
    label = id_to_label.get(predicted_index, str(predicted_index))
    probability_by_label = {
        id_to_label.get(index, str(index)): float(probability)
        for index, probability in enumerate(probabilities)
    }

    return TransformerPrediction(
        label=label,
        confidence=float(probabilities[predicted_index]),
        probabilities=probability_by_label,
    )


def predict_transformer_spam(model_dir, message, max_length=128):
    try:
        return _predict_transformer_spam_local(model_dir, message, max_length)
    except MissingTransformerDependencies:
        return _predict_transformer_spam_with_helper(model_dir, message, max_length)


def _predict_transformer_spam_with_helper(model_dir, message, max_length):
    helper_python = _find_transformer_python()
    if helper_python is None:
        raise MissingTransformerDependencies(
            "XLM-RoBERTa prediction requires torch and transformers, but the current Python environment does not "
            "have them and no compatible helper Python was found. Try running with "
            "C:\\ProgramData\\anaconda3\\envs\\py310\\python.exe ui.py."
        )

    payload = {
        "model_dir": str(Path(model_dir).resolve()),
        "message": message,
        "max_length": max_length,
    }
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    project_root = Path(model_dir).resolve().parents[1]
    hf_home = project_root / ".hf_models"
    if hf_home.exists():
        env["HF_HOME"] = str(hf_home)
        env.pop("TRANSFORMERS_CACHE", None)

    completed = subprocess.run(
        [str(helper_python), str(Path(__file__).resolve()), "--predict-json"],
        input=json.dumps(payload),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=PREDICTION_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        error_output = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"XLM-RoBERTa helper prediction failed: {error_output}")

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"XLM-RoBERTa helper returned invalid output: {completed.stdout}") from error

    return TransformerPrediction(
        label=str(result["label"]),
        confidence=float(result["confidence"]),
        probabilities={str(label): float(probability) for label, probability in result["probabilities"].items()},
    )


def _find_transformer_python():
    candidates = []
    env_python = os.environ.get("SPAM_TRANSFORMER_PYTHON")
    if env_python:
        candidates.append(Path(env_python))
    candidates.append(DEFAULT_TRANSFORMER_PYTHON)

    current_python = Path(sys.executable).resolve()
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() != current_python:
            return candidate
    return None


def _main():
    if len(sys.argv) != 2 or sys.argv[1] != "--predict-json":
        return 2

    payload = json.loads(sys.stdin.read())
    result = _predict_transformer_spam_local(
        payload["model_dir"],
        payload["message"],
        int(payload.get("max_length", 128)),
    )
    print(
        json.dumps(
            {
                "label": result.label,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
