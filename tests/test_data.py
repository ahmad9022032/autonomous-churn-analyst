"""Cleaning invariants — these tests ARE the documentation of the data audit."""

import pandas as pd
import pytest

from churn_agent import data


@pytest.fixture(scope="module")
def df():
    return data.get_dataframe()


def test_row_count_preserved(df):
    # We impute rather than drop, so counts always match the source file.
    assert len(df) == 7043


def test_no_missing_values(df):
    assert df.notna().all().all()


def test_totalcharges_numeric_with_11_imputed_zeros(df):
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    zero_tc = df[df["TotalCharges"] == 0.0]
    assert len(zero_tc) == 11
    assert (zero_tc["tenure"] == 0).all()
    assert (zero_tc["Churn"] == "No").all()


def test_seniorcitizen_normalized(df):
    assert set(df["SeniorCitizen"].unique()) == {"Yes", "No"}


def test_categorical_domains(df):
    for col, domain in data.CATEGORICAL_DOMAINS.items():
        assert set(df[col].unique()) <= domain, col


def test_service_flag_consistency(df):
    no_internet = df["InternetService"] == "No"
    for col in data.INTERNET_SUB_SERVICES:
        assert ((df[col] == "No internet service") == no_internet).all(), col
    assert (
        (df["MultipleLines"] == "No phone service") == (df["PhoneService"] == "No")
    ).all()


def test_unique_ids_and_ranges(df):
    assert df["customerID"].is_unique
    assert df["tenure"].between(0, 72).all()
    assert (df["MonthlyCharges"] > 0).all()


def test_validate_clean_catches_breakage(df):
    broken = df.copy()
    broken.loc[0, "tenure"] = 99
    with pytest.raises(AssertionError):
        data.validate_clean(broken)


def test_schema_summary_shape():
    s = data.schema_summary()
    assert s["rows"] == 7043
    assert s["churn_rate"] == pytest.approx(0.2654, abs=1e-4)
    assert s["columns"]["Contract"]["type"] == "categorical"
    assert s["columns"]["tenure"]["type"] == "numeric"
