"""Model persistence and registry with metadata, ONNX export stubs, and optional MLflow logging."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

MODELS_DIR = Path(__file__).resolve().parents[1] / 'artifacts'
REGISTRY_FILE = MODELS_DIR / 'registry.json'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelMetadata:
    name: str
    version: str
    created_at: float
    metrics: Dict[str, float]
    params: Dict[str, Any]
    notes: Optional[str] = None
    onnx_path: Optional[str] = None
    pickle_path: Optional[str] = None


def _load_registry() -> Dict[str, Any]:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {'models': []}


def _save_registry(registry: Dict[str, Any]):
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)


def save_model_version(model: Any, name: str, metrics: Dict[str, float], params: Dict[str, Any], notes: Optional[str] = None) -> ModelMetadata:
    ts = time.strftime('%Y%m%d_%H%M%S')
    pickle_path = MODELS_DIR / f"{name}_{ts}.joblib"
    joblib.dump(model, pickle_path)

    # Try ONNX export (best-effort)
    onnx_path = None
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        # NOTE: For complex pipelines, input shape may need to be adapted
        onx = convert_sklearn(model, initial_types=[('float_input', FloatTensorType([None, 10]))])
        onnx_path = str(MODELS_DIR / f"{name}_{ts}.onnx")
        with open(onnx_path, 'wb') as f:
            f.write(onx.SerializeToString())
    except Exception:
        onnx_path = None

    meta = ModelMetadata(
        name=name,
        version=ts,
        created_at=time.time(),
        metrics=metrics,
        params=params,
        notes=notes,
        onnx_path=onnx_path,
        pickle_path=str(pickle_path),
    )

    reg = _load_registry()
    reg['models'].append(asdict(meta))
    _save_registry(reg)
    return meta


def load_model(path: str):
    return joblib.load(path)


def latest_best_model() -> Optional[Dict[str, Any]]:
    reg = _load_registry()
    if not reg['models']:
        return None
    # sort by PR AUC if available, else by created_at
    def score(m):
        return m.get('metrics', {}).get('pr_auc', 0.0)
    best = sorted(reg['models'], key=lambda m: (score(m), m.get('created_at', 0)), reverse=True)[0]
    return best


def log_to_mlflow(name: str, metrics: Dict[str, float], params: Dict[str, Any], tags: Optional[Dict[str, str]] = None):
    try:
        import mlflow
        mlflow.set_experiment(name)
        with mlflow.start_run():
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
    except Exception:
        # MLflow optional; ignore if not configured
        pass
