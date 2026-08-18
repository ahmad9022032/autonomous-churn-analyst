"""Sandbox containment and semantics — the graded 'restricted execution' tool."""

import pytest

from churn_agent.sandbox import Sandbox, validate_code


@pytest.fixture(scope="module")
def box():
    sandbox = Sandbox(timeout_s=5.0)
    yield sandbox
    sandbox.close()


# ------------------------------------------------------------------ escapes
@pytest.mark.parametrize(
    "code",
    [
        "import os",
        "from os import system",
        "__import__('os')",
        "open('/etc/passwd')",
        "().__class__.__mro__",
        "df.__class__",
        "getattr(df, 'to_csv')",
        "df.to_csv('x.csv')",
        "df.to_pickle('x.pkl')",
        "pd.eval('1+1')",
        "df.eval('tenure + 1')",
        "eval('1+1')",
        "exec('x=1')",
        "globals()",
        "type(df)",
        "for i in range(3): x = i",
        "while True: x = 1",
        "def f(): return 1",
        "class A: ...",
        "_secret = 1",
        "lambda: ().__class__",
    ],
)
def test_escape_rejected_at_gate(code):
    assert validate_code(code) is not None


def test_gate_rejection_is_structured(box):
    out = box.run("import os")
    assert out["status"] == "error"
    assert "rejected" in out["result"]


def test_disallowed_pd_attr_fails_at_runtime(box):
    # read_csv is not on the pd proxy: gate passes, runtime denies.
    out = box.run("pd.read_csv('/etc/passwd')")
    assert out["status"] == "error"
    assert "AttributeError" in out["result"]


# ------------------------------------------------------------------ happy paths
def test_scalar_result_with_fact(box):
    out = box.run("(df['Churn'] == 'Yes').mean()")
    assert out["status"] == "ok"
    assert out["facts"][0]["value"] == pytest.approx(0.2654, abs=1e-3)


def test_groupby_series_facts_labeled_by_index(box):
    out = box.run(
        "df.groupby('Contract')['Churn'].apply(lambda s: (s == 'Yes').mean())"
    )
    assert out["status"] == "ok"
    labels = {f["label"]: f["value"] for f in out["facts"]}
    assert labels["Month-to-month"] == pytest.approx(0.4271, abs=1e-3)
    assert labels["row_count"] == 3


def test_multistep_with_assignment_and_comprehension(box):
    out = box.run(
        "high = df[df['MonthlyCharges'] > 100]\n"
        "sorted([round(x) for x in high['MonthlyCharges'].head(3)])"
    )
    assert out["status"] == "ok"


def test_dataframe_render_truncated(box):
    out = box.run("df")
    assert out["status"] == "ok"
    assert "7043 rows" in out["result"]
    assert len(out["result"]) < 2200


def test_lambda_apply_allowed(box):
    out = box.run("df['tenure'].apply(lambda t: t * 12).mean()")
    assert out["status"] == "ok"


def test_query_allowed(box):
    out = box.run("len(df.query('Contract == \"Two year\" and tenure > 60'))")
    assert out["status"] == "ok"


# ------------------------------------------------------------------ semantics
def test_assignment_only_is_empty_with_hint(box):
    out = box.run("x = df['tenure'].mean()")
    assert out["status"] == "empty"
    assert "expression" in out["hint"]


def test_empty_filter_flagged(box):
    out = box.run("df[df['tenure'] > 999]")
    assert out["status"] == "empty"


def test_runtime_error_is_structured(box):
    out = box.run("df['NoSuchColumn'].mean()")
    assert out["status"] == "error"
    assert "KeyError" in out["result"]


def test_df_mutation_does_not_leak_between_calls(box):
    box.run("df = df[df['tenure'] > 50]\nlen(df)")
    out = box.run("len(df)")
    assert out["status"] == "ok"
    assert out["facts"][0]["value"] == 7043


def test_timeout_kills_and_respawns(box):
    out = box.run("sum(i for i in range(10**10))")
    assert out["status"] == "timeout"
    # worker was killed; next call must transparently respawn
    out2 = box.run("len(df)")
    assert out2["status"] == "ok"
    assert out2["facts"][0]["value"] == 7043
