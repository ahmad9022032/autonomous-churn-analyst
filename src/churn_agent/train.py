"""Reproduce the served model artifact: `python -m churn_agent.train`."""

from __future__ import annotations

import json

import joblib

from .config import ARTIFACTS_DIR, METRICS_PATH, MODEL_PATH
from .data import get_dataframe
from .model import train_model


def main() -> None:
    print("training churn model (logistic regression, 5-fold CV over C)...")
    bundle, metrics = train_model(get_dataframe())

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    print(f"saved {MODEL_PATH.name} and {METRICS_PATH.name}\n")
    width = max(len(k) for k in metrics)
    for key, value in metrics.items():
        print(f"  {key:<{width}}  {value}")


if __name__ == "__main__":
    main()
