"""The numeric-provenance verifier — planted hallucinations must not survive."""

import pytest

from churn_agent.verify import FactLedger, degrade, extract_numbers, verify_draft


def ledger_with(*values):
    ledger = FactLedger()
    ledger.add([{"label": f"fact_{i}", "value": v} for i, v in enumerate(values)], "test", 0)
    return ledger


# ------------------------------------------------------------------ extraction
def test_extracts_percent_currency_thousands():
    ms = extract_numbers("Churn is 42.7%, ARPU $70.35, base 7,043 customers.")
    assert [m.value for m in ms] == [42.7, 70.35, 7043.0]
    assert ms[0].is_percent and ms[1].raw.startswith("$")


def test_customer_ids_and_code_spans_ignored():
    ms = extract_numbers("Customer 7590-VHVEG scored high; see `df.head(20)` for details.")
    assert ms == []


def test_ordinals_and_top_n_ignored():
    text = "1. First finding\n2) Second finding\nThe top 5 customers stand out."
    assert extract_numbers(text) == []


# ------------------------------------------------------------------ matching
def test_planted_hallucination_caught():
    report = verify_draft(
        "The churn rate is 42.7% and total revenue is $12,345.", ledger_with(0.427), "q"
    )
    assert report.matched == ["42.7%"]
    assert report.unmatched == ["$12,345"]
    assert not report.ok


def test_percent_form_of_fraction_matches():
    assert verify_draft("About 26.5% churn.", ledger_with(0.26537), "q").ok
    assert verify_draft("About 27% churn.", ledger_with(0.26537), "q").ok


def test_complement_matches():
    assert verify_draft("57.3% of customers stay.", ledger_with(0.427), "q").ok


def test_rounded_and_thousands_matches():
    assert verify_draft("There are 7,043 customers.", ledger_with(7043.0), "q").ok
    assert verify_draft("Average tenure is 32.4 months.", ledger_with(32.37114), "q").ok


def test_absolute_value_matches_negative_delta():
    assert verify_draft("Risk drops by 0.28.", ledger_with(-0.2827), "q").ok
    assert verify_draft("A drop of 28 percentage points.", ledger_with(-0.2827), "q").ok


def test_derived_ratio_matches():
    # 1869 / 7043 = 26.54% — quoted share derived from two computed counts
    assert verify_draft("That is 26.5% of the base.", ledger_with(1869.0, 7043.0), "q").ok


def test_question_numbers_whitelisted():
    report = verify_draft(
        "Here are the 3 customers with tenure under 12 months.",
        ledger_with(),
        "show me 3 customers with tenure under 12",
    )
    assert report.ok and report.whitelisted == 2


def test_close_but_wrong_number_rejected():
    # 0.55 displayed against a 0.427 fact: neither rounding nor 0.5% tolerance saves it
    assert not verify_draft("Churn rate is 55%.", ledger_with(0.427), "q").ok


def test_no_numbers_is_trivially_ok():
    report = verify_draft("Month-to-month contracts churn the most.", ledger_with(0.427), "q")
    assert report.ok and report.total == 0


# ------------------------------------------------------------------ degradation
def test_degrade_strips_only_bad_sentences():
    ledger = ledger_with(0.427)
    draft = "Month-to-month churn is 42.7%. Fabricated revenue is $99,999. Contracts matter."
    report = verify_draft(draft, ledger, "q")
    out = degrade(draft, report, ledger)
    assert "42.7%" in out
    assert "$99,999" not in out
    assert "removed" in out


def test_degrade_falls_back_to_facts_only():
    ledger = ledger_with(0.427, 7043.0)
    draft = "Revenue is $1,234. Profit is $5,678. Growth is 99%."
    report = verify_draft(draft, ledger, "q")
    out = degrade(draft, report, ledger)
    assert "$1,234" not in out and "99%" not in out
    assert "0.427" in out  # ledger-rendered fact
    assert "could not verify" in out


def test_degraded_output_contains_no_unverified_numbers():
    ledger = ledger_with(10.0)
    draft = "We have 10 items worth $77 total."
    report = verify_draft(draft, ledger, "q")
    out = degrade(draft, report, ledger)
    final_report = verify_draft(out, ledger, "q")
    assert final_report.ok


# ------------------------------------------------------------------ ledger
def test_ledger_render_dedupes_and_labels():
    ledger = FactLedger()
    ledger.add([{"label": "churn_rate", "value": 0.427}], "run_python", 1)
    ledger.add([{"label": "churn_rate", "value": 0.427}], "run_python", 2)
    text = ledger.render()
    assert text.count("churn_rate") == 1
    assert "[run_python]" in text
