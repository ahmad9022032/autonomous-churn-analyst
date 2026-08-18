"""Load, audit, and clean the Telco customer-churn dataset.

The cleaning decisions here are deliberate and documented (see DATA_NOTES).
`clean()` asserts the *structure* of every issue it fixes — if the input file
changes shape, we want a loud failure, not a silent mis-clean.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .config import DATA_CSV

TARGET = "Churn"
ID_COL = "customerID"

# Columns whose value depends on having internet service at all.
INTERNET_SUB_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_DOMAINS: dict[str, set[str]] = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {"Yes", "No"},  # after cleaning (raw file uses 0/1)
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    **{c: {"Yes", "No", "No internet service"} for c in INTERNET_SUB_SERVICES},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}

# Findings from the data audit, surfaced to the agent via get_data_overview
# and reproduced in the README. "Checked clean" documents the null results —
# knowing what *isn't* wrong is part of understanding the dataset.
DATA_NOTES: dict[str, list[str]] = {
    "issues_found_and_fixed": [
        "TotalCharges was read as text: 11 rows contain a single space instead of "
        "a number. All 11 are tenure=0 customers (never billed yet, all Churn=No). "
        "Fixed by imputing 0.0 — semantically the true amount billed so far — and "
        "keeping the rows so counts always match the source file.",
        "SeniorCitizen was encoded 0/1 while every other binary column uses Yes/No. "
        "Normalized to Yes/No for consistency.",
    ],
    "modeling_notes": [
        "customerID is a pure identifier — excluded from model features, kept for lookups.",
        "Class imbalance: 26.54% churn — shaped the evaluation-metric choice.",
        "TotalCharges is nearly tenure x MonthlyCharges (collinear) — excluded from "
        "model features, retained in the dataframe for EDA queries.",
    ],
    "checked_clean": [
        "No duplicate rows or duplicate customer IDs.",
        "No stray whitespace or casing inconsistencies outside TotalCharges.",
        "No negative or out-of-range numeric values.",
        "Service-flag structure exactly consistent (e.g. 'No internet service' "
        "appears iff InternetService == 'No').",
        "TotalCharges is consistent with tenure x MonthlyCharges for every row "
        "(no injected corruption).",
    ],
}


def load_raw() -> pd.DataFrame:
    """Read the CSV exactly as shipped (TotalCharges arrives as object dtype)."""
    return pd.read_csv(DATA_CSV)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning decisions; assert the structure they rely on."""
    df = raw.copy()

    # Issue 1: TotalCharges is text with 11 single-space blanks (tenure-0 customers).
    blank = df["TotalCharges"].astype(str).str.strip() == ""
    assert blank.sum() == 11, f"expected 11 blank TotalCharges, got {blank.sum()}"
    assert (df.loc[blank, "tenure"] == 0).all(), "blank TotalCharges outside tenure=0"
    assert (df.loc[blank, TARGET] == "No").all(), "blank TotalCharges with churn=Yes"
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Issue 2: SeniorCitizen is 0/1; every other binary is Yes/No.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    assert df["SeniorCitizen"].notna().all(), "unexpected SeniorCitizen values"

    df["tenure"] = df["tenure"].astype("int64")
    df["MonthlyCharges"] = df["MonthlyCharges"].astype("float64")
    return df


def validate_clean(df: pd.DataFrame) -> None:
    """Invariants the rest of the system relies on. Raises AssertionError on breach."""
    assert len(df) == 7043, f"row count changed: {len(df)}"
    assert df.notna().all().all(), "NaNs present after cleaning"
    assert df[ID_COL].is_unique, "duplicate customer IDs"
    for col, domain in CATEGORICAL_DOMAINS.items():
        extra = set(df[col].unique()) - domain
        assert not extra, f"{col} has unexpected values: {extra}"
    assert df["tenure"].between(0, 72).all(), "tenure out of range"
    assert (df["MonthlyCharges"] > 0).all(), "non-positive MonthlyCharges"
    assert (df["TotalCharges"] >= 0).all(), "negative TotalCharges"
    no_phone = df["PhoneService"] == "No"
    assert ((df["MultipleLines"] == "No phone service") == no_phone).all()
    no_internet = df["InternetService"] == "No"
    for col in INTERNET_SUB_SERVICES:
        assert ((df[col] == "No internet service") == no_internet).all(), col


@lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    """The canonical cleaned dataframe (cached). Callers must not mutate it."""
    df = clean(load_raw())
    validate_clean(df)
    return df


def schema_summary() -> dict:
    """Compact machine-readable schema for the get_data_overview tool."""
    df = get_dataframe()
    columns = {}
    for col in df.columns:
        if col == ID_COL:
            columns[col] = {"type": "id", "example": df[col].iloc[0]}
        elif pd.api.types.is_numeric_dtype(df[col]):
            columns[col] = {
                "type": "numeric",
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": round(float(df[col].mean()), 4),
            }
        else:
            counts = df[col].value_counts()
            columns[col] = {
                "type": "categorical",
                "values": {str(k): int(v) for k, v in counts.items()},
            }
    return {
        "rows": len(df),
        "columns": columns,
        "churn_rate": round(float((df[TARGET] == "Yes").mean()), 4),
        "notes": DATA_NOTES,
    }
