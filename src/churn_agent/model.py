"""Churn model as a callable: training, prediction, explanation, normalization.

The served artifact is a plain dict (joblib) so the notebook and this package
can produce/consume it interchangeably. `ChurnModel` wraps it with the
operations the agent tools need. Mirrors the notebook's modeling decisions:
logistic regression, TotalCharges and customerID excluded, no class_weight
(calibration is part of the product — see notebook section 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_PATH, RANDOM_STATE
from .data import ID_COL, INTERNET_SUB_SERVICES, TARGET, get_dataframe

NUMERIC = ["tenure", "MonthlyCharges"]
EXCLUDED = [ID_COL, "TotalCharges", TARGET]


def _feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [c for c in df.columns if c not in NUMERIC + EXCLUDED]
    return categorical, NUMERIC


def train_model(df: pd.DataFrame) -> tuple[dict, dict]:
    """Train on a stratified 80% split, evaluate on the 20% holdout.

    Returns (bundle, metrics). The served pipeline is the train-split fit, so
    the reported metrics describe exactly the artifact being served.
    """
    categorical, numeric = _feature_cols(df)
    X = df[categorical + numeric]
    y = (df[TARGET] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )
    grid = GridSearchCV(
        Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=2000))]),
        {"clf__C": [0.1, 1.0, 10.0]},
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    ).fit(X_train, y_train)
    pipeline = grid.best_estimator_

    proba = pipeline.predict_proba(X_test)[:, 1]
    k = max(1, int(0.10 * len(y_test)))
    top_idx = np.argsort(proba)[::-1][:k]
    metrics = {
        "PR-AUC": round(float(average_precision_score(y_test, proba)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_test, proba)), 4),
        "Brier": round(float(brier_score_loss(y_test, proba)), 4),
        "recall@top-10%": round(float(y_test.iloc[top_idx].sum() / y_test.sum()), 4),
        "precision@top-10%": round(float(y_test.iloc[top_idx].mean()), 4),
        "lift@top-10%": round(float(y_test.iloc[top_idx].mean() / y_test.mean()), 2),
        "churn base rate": round(float(y_test.mean()), 4),
        "best_C": grid.best_params_["clf__C"],
    }

    prep = pipeline.named_steps["prep"]
    Z = prep.transform(X_train)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    bundle = {
        "pipeline": pipeline,
        "categorical": categorical,
        "numeric": numeric,
        "feature_names": list(prep.get_feature_names_out()),
        "z_mean": Z.mean(axis=0),
        "defaults": {c: X[c].mode()[0] for c in categorical}
        | {c: float(X[c].median()) for c in numeric},
        "allowed_values": {c: sorted(X[c].unique()) for c in categorical},
        "train_scores_sorted": np.sort(pipeline.predict_proba(X_train)[:, 1]),
        "metrics": metrics,
    }
    return bundle, metrics


@dataclass
class NormalizedCustomer:
    row: pd.DataFrame
    applied_defaults: dict[str, Any] = field(default_factory=dict)
    fixes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class ChurnModel:
    """Operations over a trained bundle: predict, explain, normalize, percentile."""

    def __init__(self, bundle: dict):
        self.b = bundle
        self.features: list[str] = bundle["categorical"] + bundle["numeric"]

    # ---------------------------------------------------------- prediction
    def predict(self, row: pd.DataFrame) -> float:
        return float(self.b["pipeline"].predict_proba(row[self.features])[0, 1])

    def percentile(self, score: float) -> float:
        """Share of training customers scoring below `score` (risk context)."""
        arr = self.b["train_scores_sorted"]
        return round(float(np.searchsorted(arr, score) / len(arr)), 4)

    # ---------------------------------------------------------- explanation
    def top_factors(self, row: pd.DataFrame, k: int = 4) -> list[dict]:
        """Exact linear contributions vs. the training mean, per original column."""
        prep = self.b["pipeline"].named_steps["prep"]
        clf = self.b["pipeline"].named_steps["clf"]
        z = prep.transform(row[self.features])
        z = z.toarray()[0] if hasattr(z, "toarray") else np.asarray(z)[0]
        contrib = clf.coef_[0] * (z - self.b["z_mean"])

        per_column: dict[str, float] = {}
        for name, value in zip(self.b["feature_names"], contrib):
            if name.startswith("num__"):
                parent = name[len("num__") :]
            else:  # cat__<column>_<level>; column names contain no underscores
                parent = name[len("cat__") :].split("_", 1)[0]
            per_column[parent] = per_column.get(parent, 0.0) + float(value)

        ranked = sorted(per_column.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return [
            {
                "feature": col,
                "customer_value": row.iloc[0][col]
                if not isinstance(row.iloc[0][col], np.generic)
                else row.iloc[0][col].item(),
                "direction": "increases risk" if val > 0 else "decreases risk",
                "log_odds": round(val, 3),
            }
            for col, val in ranked[:k]
        ]

    # ---------------------------------------------------------- normalization
    def normalize(self, attributes: dict[str, Any]) -> NormalizedCustomer:
        """Build a full, structurally consistent customer row from a partial dict.

        Unknown columns/values become errors (fed back to the agent's
        self-check); filled defaults and consistency fixes are reported so the
        final answer can disclose its assumptions.
        """
        out = NormalizedCustomer(row=pd.DataFrame())
        attrs: dict[str, Any] = {}

        for key, value in attributes.items():
            match = next((c for c in self.features if c.lower() == str(key).lower()), None)
            if match is None:
                out.errors.append(
                    f"Unknown column {key!r}. Valid columns: {', '.join(self.features)}"
                )
                continue
            attrs[match] = value

        # convenience coercions before domain validation
        if "SeniorCitizen" in attrs and attrs["SeniorCitizen"] in (0, 1, True, False, "0", "1"):
            attrs["SeniorCitizen"] = "Yes" if attrs["SeniorCitizen"] in (1, True, "1") else "No"
            out.fixes.append("SeniorCitizen coerced to Yes/No encoding")

        for col in self.b["categorical"]:
            if col not in attrs:
                continue
            allowed = self.b["allowed_values"][col]
            match = next((a for a in allowed if a.lower() == str(attrs[col]).lower()), None)
            if match is None:
                out.errors.append(
                    f"Invalid value {attrs[col]!r} for {col}. Allowed: {allowed}"
                )
            elif match != attrs[col]:
                out.fixes.append(f"{col}: {attrs[col]!r} matched to {match!r}")
                attrs[col] = match
            else:
                attrs[col] = match

        for col in self.b["numeric"]:
            if col not in attrs:
                continue
            try:
                attrs[col] = float(attrs[col])
            except (TypeError, ValueError):
                out.errors.append(f"{col} must be numeric, got {attrs[col]!r}")

        if out.errors:
            return out

        full = dict(self.b["defaults"])
        full.update(attrs)
        out.applied_defaults = {
            c: self.b["defaults"][c] for c in self.features if c not in attrs
        }

        # structural consistency: service parents govern their dependents
        if full["PhoneService"] == "No":
            if attrs.get("MultipleLines") not in (None, "No phone service"):
                out.fixes.append("MultipleLines forced to 'No phone service' (PhoneService=No)")
            full["MultipleLines"] = "No phone service"
        elif full["MultipleLines"] == "No phone service":
            out.fixes.append("MultipleLines 'No phone service' reset to 'No' (PhoneService=Yes)")
            full["MultipleLines"] = "No"
        if full["InternetService"] == "No":
            for col in INTERNET_SUB_SERVICES:
                if attrs.get(col) not in (None, "No internet service"):
                    out.fixes.append(f"{col} forced to 'No internet service' (InternetService=No)")
                full[col] = "No internet service"
        else:
            for col in INTERNET_SUB_SERVICES:
                if full[col] == "No internet service":
                    out.fixes.append(f"{col} 'No internet service' reset to 'No' (has internet)")
                    full[col] = "No"

        # applied_defaults may have been overridden by consistency fixes; re-sync
        out.applied_defaults = {
            c: full[c] for c in self.features if c not in attrs
        }
        out.row = pd.DataFrame([full])[self.features]
        return out


# --------------------------------------------------------------- module API
@lru_cache(maxsize=1)
def get_model() -> ChurnModel:
    """Load the served bundle; retrain transparently if the artifact is unusable."""
    try:
        bundle = joblib.load(MODEL_PATH)
        assert {"pipeline", "categorical", "numeric"} <= bundle.keys()
    except Exception:
        bundle, _ = train_model(get_dataframe())
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
    return ChurnModel(bundle)


def predict_churn_risk(customer_id: str) -> dict:
    """The brief's requested callable: predict_churn_risk(id) -> {risk_score, top_factors}."""
    df = get_dataframe()
    hit = df[df[ID_COL] == customer_id]
    if hit.empty:
        raise KeyError(f"customer {customer_id!r} not found")
    model = get_model()
    row = hit.iloc[[0]]
    score = model.predict(row)
    return {
        "risk_score": round(score, 4),
        "risk_percentile": model.percentile(score),
        "top_factors": model.top_factors(row),
    }
