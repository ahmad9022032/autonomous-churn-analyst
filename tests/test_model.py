"""Model bundle behaviors the agent tools depend on."""

import numpy as np
import pytest

from churn_agent import model as m
from churn_agent.data import get_dataframe


@pytest.fixture(scope="module")
def quick_model():
    # Small sample keeps the suite fast while exercising the real training code.
    df = get_dataframe()
    bundle, metrics = m.train_model(df.sample(n=800, random_state=0))
    return m.ChurnModel(bundle), metrics


def test_metrics_beat_trivial_baseline(quick_model):
    _, metrics = quick_model
    assert metrics["PR-AUC"] > 0.35  # trivial baseline = base rate ~0.27
    assert metrics["ROC-AUC"] > 0.7


def test_scores_are_probabilities(quick_model):
    model, _ = quick_model
    df = get_dataframe()
    for i in [0, 5, 100]:
        score = model.predict(df.iloc[[i]])
        assert 0.0 <= score <= 1.0
        assert 0.0 <= model.percentile(score) <= 1.0


def test_top_factors_reference_real_columns(quick_model):
    model, _ = quick_model
    df = get_dataframe()
    factors = model.top_factors(df.iloc[[3]])
    assert len(factors) == 4
    for f in factors:
        assert f["feature"] in model.features
        assert f["direction"] in ("increases risk", "decreases risk")
        assert isinstance(f["log_odds"], float)


def test_normalize_fills_and_discloses_defaults(quick_model):
    model, _ = quick_model
    norm = model.normalize({"Contract": "Month-to-month", "tenure": 2})
    assert not norm.errors
    assert norm.row.iloc[0]["Contract"] == "Month-to-month"
    assert "MonthlyCharges" in norm.applied_defaults
    assert "Contract" not in norm.applied_defaults


def test_normalize_enforces_structural_consistency(quick_model):
    model, _ = quick_model
    norm = model.normalize({"InternetService": "No", "OnlineSecurity": "Yes"})
    assert not norm.errors
    assert norm.row.iloc[0]["OnlineSecurity"] == "No internet service"
    assert any("OnlineSecurity" in f for f in norm.fixes)


def test_normalize_rejects_unknown_values(quick_model):
    model, _ = quick_model
    norm = model.normalize({"Contract": "Decade-long"})
    assert norm.errors and "Contract" in norm.errors[0]
    norm2 = model.normalize({"FavouriteColor": "blue"})
    assert norm2.errors and "Unknown column" in norm2.errors[0]


def test_normalize_case_insensitive_matching(quick_model):
    model, _ = quick_model
    norm = model.normalize({"contract": "month-to-month", "SeniorCitizen": 1})
    assert not norm.errors
    assert norm.row.iloc[0]["Contract"] == "Month-to-month"
    assert norm.row.iloc[0]["SeniorCitizen"] == "Yes"


def test_bundle_roundtrip(tmp_path, quick_model):
    import joblib

    model, _ = quick_model
    path = tmp_path / "m.joblib"
    joblib.dump(model.b, path)
    reloaded = m.ChurnModel(joblib.load(path))
    df = get_dataframe()
    assert reloaded.predict(df.iloc[[7]]) == pytest.approx(model.predict(df.iloc[[7]]))


def test_predict_churn_risk_contract():
    df = get_dataframe()
    out = m.predict_churn_risk(df.iloc[0]["customerID"])
    assert set(out) == {"risk_score", "risk_percentile", "top_factors"}
    with pytest.raises(KeyError):
        m.predict_churn_risk("ZZZZ-XXXXX")
