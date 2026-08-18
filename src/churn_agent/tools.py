"""The agent's tool surface: six tools, one envelope shape, a dispatcher that never raises.

Design split: everything touching the *model* is a deterministic canned tool
(those paths are graded and must not depend on a free-tier LLM writing correct
pandas); free-form code goes through the sandboxed `run_python` only for EDA,
where a failed attempt is cheap and the self-check loop retries.

Envelope: {"status": "ok"|"error"|"empty", "data": ..., "facts": [{label, value}], "hint": ...}
`facts` is the machine-readable feed for the numeric-provenance ledger — the
verifier never parses numbers out of prose.
"""

from __future__ import annotations

import difflib
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data import ID_COL, TARGET, get_dataframe, schema_summary
from .model import get_model
from .sandbox import Sandbox

_sandbox: Sandbox | None = None


def _get_sandbox() -> Sandbox:
    global _sandbox
    if _sandbox is None:
        _sandbox = Sandbox()
    return _sandbox


def warm_up() -> None:
    """Preload dataframe, model, and sandbox worker (avoids first-call latency)."""
    get_dataframe()
    get_model()
    _get_sandbox().run("len(df)")


def _ok(data: Any, facts: list[dict], hint: str | None = None) -> dict:
    return {"status": "ok", "data": data, "facts": facts, "hint": hint}


def _error(message: str, hint: str | None = None) -> dict:
    return {"status": "error", "data": message, "facts": [], "hint": hint}


def _empty(message: str, hint: str | None = None) -> dict:
    return {"status": "empty", "data": message, "facts": [], "hint": hint}


def _fact(label: str, value: Any) -> dict:
    return {"label": str(label)[:80], "value": float(value)}


def _find_customer(customer_id: str) -> pd.DataFrame:
    df = get_dataframe()
    return df[df[ID_COL] == str(customer_id).strip()]


def _match_column(name: str) -> tuple[str | None, str | None]:
    """Case-insensitive column match; returns (match, did_you_mean_hint)."""
    df = get_dataframe()
    exact = next((c for c in df.columns if c.lower() == str(name).lower()), None)
    if exact:
        return exact, None
    close = difflib.get_close_matches(str(name), list(df.columns), n=3, cutoff=0.5)
    hint = f"column {name!r} does not exist. "
    hint += f"Closest columns: {close}. " if close else ""
    hint += f"All columns: {', '.join(df.columns)}"
    return None, hint


# ------------------------------------------------------------------- tools
def get_data_overview() -> dict:
    summary = schema_summary()
    facts = [
        _fact("total_rows", summary["rows"]),
        _fact("churn_rate", summary["churn_rate"]),
    ]
    for col, info in summary["columns"].items():
        if info["type"] == "categorical":
            for value, count in info["values"].items():
                facts.append(_fact(f"count[{col}={value}]", count))
        elif info["type"] == "numeric":
            for stat in ("min", "max", "mean"):
                facts.append(_fact(f"{col}_{stat}", info[stat]))
    return _ok(summary, facts)


def run_python(code: str) -> dict:
    env = _get_sandbox().run(code)
    return {
        "status": env["status"] if env["status"] != "timeout" else "error",
        "data": env["result"],
        "facts": env["facts"],
        "hint": env["hint"],
    }


def _predict_payload(row: pd.DataFrame, extra: dict | None = None) -> tuple[dict, list[dict]]:
    model = get_model()
    score = model.predict(row)
    percentile = model.percentile(score)
    factors = model.top_factors(row)
    data = {
        "risk_score": round(score, 4),
        "risk_percentile": percentile,
        "top_factors": factors,
    } | (extra or {})
    facts = [_fact("risk_score", round(score, 4)), _fact("risk_percentile", percentile)]
    facts += [_fact(f"log_odds[{f['feature']}]", f["log_odds"]) for f in factors]
    return data, facts


def predict_churn(customer_id: str) -> dict:
    hit = _find_customer(customer_id)
    if hit.empty:
        return _error(
            f"customer {customer_id!r} not found",
            hint="IDs look like 7590-VHVEG (4 digits, dash, 5 letters); "
            "use run_python to search if unsure",
        )
    row = hit.iloc[[0]]
    snapshot = {
        c: (row.iloc[0][c].item() if isinstance(row.iloc[0][c], np.generic) else row.iloc[0][c])
        for c in ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "InternetService", "PaymentMethod", TARGET]
    }
    data, facts = _predict_payload(row, {"customer_id": customer_id, "snapshot": snapshot})
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        facts.append(_fact(f"snapshot_{col}", snapshot[col]))
    return _ok(data, facts)


def predict_hypothetical(attributes: dict) -> dict:
    if not isinstance(attributes, dict) or not attributes:
        return _error(
            "attributes must be a non-empty object of column -> value",
            hint='example: {"Contract": "Month-to-month", "tenure": 2, "InternetService": "Fiber optic"}',
        )
    norm = get_model().normalize(attributes)
    if norm.errors:
        return _error("; ".join(norm.errors), hint="fix the listed values and retry")
    data, facts = _predict_payload(
        norm.row,
        {
            "provided": attributes,
            "applied_defaults": norm.applied_defaults,
            "consistency_fixes": norm.fixes,
        },
    )
    for col in ("tenure", "MonthlyCharges"):
        facts.append(_fact(f"input_{col}", float(norm.row.iloc[0][col])))
    return _ok(
        data,
        facts,
        hint="mention the applied_defaults when presenting this result — the "
        "user did not specify those fields",
    )


def what_if(customer_id: str, changes: dict) -> dict:
    hit = _find_customer(customer_id)
    if hit.empty:
        return _error(f"customer {customer_id!r} not found")
    if not isinstance(changes, dict) or not changes:
        return _error(
            "changes must be a non-empty object of column -> new value",
            hint='example: {"Contract": "Two year"}',
        )
    model = get_model()
    row = hit.iloc[[0]]
    baseline = model.predict(row)

    current = {c: row.iloc[0][c] for c in model.features}
    norm = model.normalize({**current, **changes})
    if norm.errors:
        return _error("; ".join(norm.errors), hint="fix the listed values and retry")
    projected = model.predict(norm.row)
    delta = round(projected - baseline, 4)

    data = {
        "customer_id": customer_id,
        "changes": changes,
        "consistency_fixes": norm.fixes,
        "baseline_risk": round(baseline, 4),
        "projected_risk": round(projected, 4),
        "delta": delta,
        "baseline_top_factors": model.top_factors(row),
        "projected_top_factors": model.top_factors(norm.row),
    }
    facts = [
        _fact("baseline_risk", round(baseline, 4)),
        _fact("projected_risk", round(projected, 4)),
        _fact("delta", delta),
        _fact("baseline_percentile", model.percentile(baseline)),
        _fact("projected_percentile", model.percentile(projected)),
    ]
    return _ok(data, facts)


def segment_risk(group_by: str | None = None, filter: str | None = None) -> dict:
    df = get_dataframe()
    model = get_model()

    if filter:
        try:
            df = df.query(filter)
        except Exception as exc:
            return _error(
                f"invalid filter {filter!r}: {exc}",
                hint="use pandas query syntax over these columns: "
                + ", ".join(get_dataframe().columns),
            )
        if df.empty:
            return _empty(
                f"no customers match filter {filter!r}",
                hint="check category spellings via get_data_overview",
            )

    scores = model.b["pipeline"].predict_proba(df[model.features])[:, 1]
    observed = (df[TARGET] == "Yes").astype(float)

    if group_by is None:
        data = {
            "segment": "all selected customers",
            "n": int(len(df)),
            "avg_predicted_risk": round(float(scores.mean()), 4),
            "observed_churn_rate": round(float(observed.mean()), 4),
        }
        facts = [
            _fact("n", len(df)),
            _fact("avg_predicted_risk", data["avg_predicted_risk"]),
            _fact("observed_churn_rate", data["observed_churn_rate"]),
        ]
        return _ok(data, facts)

    col, hint = _match_column(group_by)
    if col is None:
        return _error(f"cannot group by {group_by!r}", hint=hint)
    if col == ID_COL:
        return _error("grouping by customerID is meaningless (one row per group)")

    keys: pd.Series = df[col]
    if pd.api.types.is_numeric_dtype(keys):
        keys = pd.qcut(keys, q=4, duplicates="drop").astype(str)

    grouped = (
        pd.DataFrame({"segment": keys, "risk": scores, "observed": observed})
        .groupby("segment", observed=True)
        .agg(n=("risk", "size"), avg_predicted_risk=("risk", "mean"), observed_churn_rate=("observed", "mean"))
        .round(4)
        .sort_values("avg_predicted_risk", ascending=False)
        .reset_index()
    )
    records = grouped.to_dict(orient="records")
    facts = []
    for r in records:
        facts.append(_fact(f"n[{r['segment']}]", r["n"]))
        facts.append(_fact(f"avg_risk[{r['segment']}]", r["avg_predicted_risk"]))
        facts.append(_fact(f"observed_churn[{r['segment']}]", r["observed_churn_rate"]))
    return _ok({"group_by": col, "filter": filter, "segments": records}, facts)


# ------------------------------------------------------------------ registry
_DF_COLUMNS = (
    "customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, "
    "MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, "
    "TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, "
    "PaymentMethod, MonthlyCharges, TotalCharges, Churn"
)

TOOLS: dict[str, dict] = {
    "get_data_overview": {
        "fn": get_data_overview,
        "schema": {
            "name": "get_data_overview",
            "description": "Schema, allowed category values, numeric ranges, churn base "
            "rate, and data-cleaning notes for the customer dataset. Call this before "
            "referencing any column you are not certain exists.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "run_python": {
        "fn": run_python,
        "schema": {
            "name": "run_python",
            "description": "Run restricted pandas code against the cleaned dataframe "
            "`df` (7043 customers; columns: " + _DF_COLUMNS + "). pd and np are "
            "available; imports, loops and file access are blocked. The value of the "
            "LAST EXPRESSION is returned. Examples: "
            "df['MonthlyCharges'].describe() | "
            "df.groupby('Contract')['Churn'].apply(lambda s: (s=='Yes').mean()) | "
            "df[df['tenure']<6].shape[0]",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "pandas code; end with an expression"}},
                "required": ["code"],
            },
        },
    },
    "predict_churn": {
        "fn": predict_churn,
        "schema": {
            "name": "predict_churn",
            "description": "Model-predicted churn risk for one existing customer by ID: "
            "risk score (0-1 probability), percentile vs all customers, top factors "
            "pushing risk up or down, and a snapshot of the customer.",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string", "description": "e.g. 7590-VHVEG"}},
                "required": ["customer_id"],
            },
        },
    },
    "predict_hypothetical": {
        "fn": predict_hypothetical,
        "schema": {
            "name": "predict_hypothetical",
            "description": "Model churn risk for a NEW hypothetical customer described "
            "by any subset of dataset columns; unspecified fields are filled with "
            "training-data defaults and reported back as applied_defaults.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attributes": {
                        "type": "object",
                        "description": 'column -> value, e.g. {"Contract": "Month-to-month", '
                        '"tenure": 2, "MonthlyCharges": 95, "InternetService": "Fiber optic", '
                        '"SeniorCitizen": "Yes"}',
                    }
                },
                "required": ["attributes"],
            },
        },
    },
    "what_if": {
        "fn": what_if,
        "schema": {
            "name": "what_if",
            "description": "Project how an EXISTING customer's churn risk would change "
            "under modified feature values: baseline risk, projected risk, and delta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "e.g. 7590-VHVEG"},
                    "changes": {
                        "type": "object",
                        "description": 'column -> new value, e.g. {"Contract": "Two year"}',
                    },
                },
                "required": ["customer_id", "changes"],
            },
        },
    },
    "segment_risk": {
        "fn": segment_risk,
        "schema": {
            "name": "segment_risk",
            "description": "Aggregate model-predicted churn risk and observed churn rate "
            "across customer segments. Optional group_by column (numeric columns are "
            "binned into quartiles) and optional pandas-query filter. With neither, "
            "returns the overall aggregate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string", "description": "column to segment by, e.g. Contract"},
                    "filter": {"type": "string", "description": "pandas query, e.g. InternetService == 'Fiber optic'"},
                },
                "required": [],
            },
        },
    },
}

TOOL_SCHEMAS = [
    {"type": "function", "function": t["schema"]} for t in TOOLS.values()
]


def dispatch(name: str, args: dict) -> dict:
    """Resolve (fuzzily) and execute a tool. Always returns an envelope."""
    resolved = name if name in TOOLS else None
    if resolved is None:
        lower = {t.lower(): t for t in TOOLS}
        resolved = lower.get(str(name).lower())
    if resolved is None:
        close = difflib.get_close_matches(str(name), list(TOOLS), n=1, cutoff=0.6)
        if close:
            resolved = close[0]
    if resolved is None:
        return _error(
            f"unknown tool {name!r}",
            hint=f"available tools: {', '.join(TOOLS)}",
        )

    fn: Callable = TOOLS[resolved]["fn"]
    params = TOOLS[resolved]["schema"]["parameters"]
    known = set(params["properties"])
    required = set(params["required"])
    args = args if isinstance(args, dict) else {}
    missing = required - set(args)
    if missing:
        return _error(
            f"missing required argument(s) for {resolved}: {', '.join(sorted(missing))}",
            hint=str(params),
        )
    try:
        return fn(**{k: v for k, v in args.items() if k in known})
    except Exception as exc:  # tools must never raise into the loop
        return _error(f"{type(exc).__name__}: {exc}", hint="adjust arguments and retry")
