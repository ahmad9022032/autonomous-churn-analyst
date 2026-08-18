# Autonomous Churn Analyst

An autonomous data-analyst agent over the Telco customer-churn dataset (7,043 customers).
You ask questions in natural language; the agent **plans**, **computes real answers with
tools** (a trained churn model + restricted pandas execution), **self-checks**, and — the
core design goal — **never states a number it didn't actually compute**: every figure in
every answer is verified against a ledger of computed results before it ships.

> **Hosted app:** _fill in after deploy (Streamlit Community Cloud)_
> **Notebook:** [`notebooks/churn_eda_and_model.ipynb`](notebooks/churn_eda_and_model.ipynb) (Colab-compatible)

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[app,dev]"
cp .env.example .env            # add your free Groq key (console.groq.com)

streamlit run app/streamlit_app.py     # chat UI
python -m churn_agent.cli              # terminal REPL with live verification trace
python -m churn_agent.cli "average churn risk by payment method"   # one-shot
python -m churn_agent.train            # reproduce the model artifact
pytest                                 # 76 tests, all offline (no API key needed)
```

## What I built

| Stage | Deliverable |
|---|---|
| 1 — Model as a callable tool | Cleaning pipeline with documented decisions (`data.py`), logistic-regression churn model with per-customer explanations (`model.py`), committed artifact + metrics, `predict_churn_risk(customer_id) -> {risk_score, risk_percentile, top_factors}` |
| 2 — Chat interface | Streamlit chat app (`app/streamlit_app.py`) rendering the agent's live plan/tool/verification trace; CLI REPL (`cli.py`) with the same events |
| 3 — The agent | Hand-rolled plan-act-check loop (`agent.py`) with six tools, a restricted execution sandbox, deterministic self-checks, and a numeric-provenance verifier (`verify.py`) |

```
question ──► LLM plans ──► tool rounds (≤8) ──► draft answer
                              │                     │
                              ▼                     ▼
                       6 tools, each returns   every number in the draft
                       {status, data, facts}   must match a ledger fact
                              │                (rounding-aware transforms)
                              ▼                     │
                       facts accumulate        unverified → revise (≤2)
                       in a FactLedger         still bad → strip claims /
                                               facts-only fallback answer
```

**Tools:** `get_data_overview` · `run_python` (sandboxed pandas) · `predict_churn` ·
`predict_hypothetical` · `what_if` · `segment_risk`. Everything touching the *model* is a
deterministic canned tool (those paths are graded and must not depend on a free-tier LLM
writing correct pandas); free-form code is allowed only for EDA, where a failed attempt is
cheap and the self-check loop retries. The model tools read the dataset singleton — the
"model tool invokes the dataset tool" behavior from the brief.

## Data issues found (and the audit that found nothing else)

The brief deliberately doesn't say what's wrong with the file. My audit
(notebook §2, executable asserts in `data.py`):

1. **`TotalCharges` loads as text** — 11 values are a single space. All 11 belong to
   `tenure = 0` customers with `Churn = "No"`: brand-new, never-billed customers. This is a
   *structural* blank, not corruption. **Fix: impute `0.0`** (the semantically true amount
   billed so far — not a statistical guess) **and keep the rows**, so every count the agent
   computes matches the source file. Dropping 11/7043 rows would also have been defensible
   for pure modeling, but silently desynchronizing counts is poison in a system graded on
   numeric traceability.
2. **`SeniorCitizen` is encoded 0/1** while every other binary column is Yes/No.
   Normalized to Yes/No so the agent (and users) see one consistent encoding.
3. **Null-result audit** — checked and found *clean*: no duplicate rows or IDs, no stray
   whitespace or casing outside TotalCharges, no negative/impossible values, service-flag
   structure exactly consistent (`"No internet service"` appears iff
   `InternetService == "No"`), and `TotalCharges ≈ tenure × MonthlyCharges` for every row.
   This file is the pristine classic Telco dataset; documenting what *isn't* wrong is part
   of understanding it.

Modeling-relevant observations: 26.54% churn (class imbalance), `customerID` is a pure
identifier (excluded from features), and `TotalCharges` is nearly `tenure ×
MonthlyCharges` (collinear — **excluded from model features**, both to keep the linear
coefficients stable/explainable and because a what-if that changes `tenure` while holding
a stale `TotalCharges` would describe a physically impossible customer).

## Why this metric

- **Rejected: accuracy.** Predicting "nobody churns" scores 73.5%. Any single-threshold
  metric at 0.5 inherits milder versions of the same blindness.
- **Primary: PR-AUC** — the business question is "how well are the churners (the minority
  we care about) ranked?", and PR-AUC gives no credit for confidently ranking easy
  non-churners, unlike ROC-AUC under imbalance.
- **Secondary: ROC-AUC** — comparability anchor (it's the lingua franca for this dataset).
- **Operational: recall/precision@top-decile and lift** — a retention team has a contact
  budget, not a threshold: "if we can call 10% of the base, what share of churners is on
  the list?" ties the metric to the decision the model actually supports.
- **Calibration: Brier + reliability table** — the agent quotes probabilities to end
  users, so the numbers must mean what they say. This is also why there is **no
  `class_weight="balanced"`**: reweighting inflates predicted probabilities and wrecks
  calibration; imbalance is handled by stratification, minority-focused ranking metrics,
  and reporting percentile context next to each score.

Holdout (stratified 20%): **PR-AUC 0.6305 · ROC-AUC 0.8395 · Brier 0.1389 ·
precision@top-decile 0.736 (lift 2.77)**. Logistic regression beat
HistGradientBoosting on CV PR-AUC (0.660 vs 0.649) while being exactly explainable
per-customer (coefficient × deviation-from-training-mean, one-hot terms folded back to
their parent column) — so the interpretable model wasn't even a trade-off here.

## How planning & verification work (the part the brief cares about most)

**Planning** is prompt-induced: the first assistant turn must open with a one-line
`PLAN: … -> …`, emitted as an event so both UIs show it. A separate planning LLM call
would cost +1 rate-limited request per question for marginal gain.

**Self-checks are deterministic code, not vibes.** Every tool returns
`{status: ok|error|empty, data, facts, hint}`. Errors and empty results are fed back with
targeted hints ("column names are…", "end with an expression"); sanity rules flag
suspicious values (rates > 100, negative counts, NaN); a tool failing repeatedly escalates
to "try another approach or answer honestly with what you have". Rate-limit design: tool
results are truncated envelopes, per-question tool traces are dropped from memory (only
(question, answer) pairs are kept — which is also what makes follow-up questions work),
and there's a hard cap of 12 LLM calls per question.

**The numeric-provenance verifier** is the centerpiece. Every tool result contributes
machine-readable `facts` to a per-question ledger (the verifier never parses numbers out
of prose). When the model produces a draft, every figure in it (percents, currency,
thousands separators; customer IDs and code spans excluded; ordinals, "top N", and numbers
from the user's question whitelisted) must match a ledger fact under a small transform
cascade — identity, ×100/÷100, complement (100−x), absolute value, and pairwise
derivations (ratio/difference/sum/share) of two facts. Matching is rounding-aware:
"26.5%" verifies against a computed 0.26537.

Unverified figures trigger a revision (≤2) that shows the model exactly which figures
failed and the rendered fact ledger. If revision fails, **honest degradation**: sentences
containing unverified figures are stripped (with a disclosure note); if most of the answer
was unverifiable, the reply is rebuilt purely from ledger facts — a path that cannot
hallucinate by construction. The CLI and Streamlit both display the verdict, e.g.
"7/7 figures verified against computed results".

**A false-accept caught during live testing** (kept in the git history): the model claimed
"month-to-month is ≈ 38% of all customers" (wrong — it's 55%), yet "38" verified, because
with dozens of ledger facts there are ~16k pairwise-derived candidates and an unrelated
ratio (1473/3875 = 38.0%) collided with it. Fix: pairwise-derived matches now require ≥3
significant digits in the displayed figure — *precision earns trust*; vague figures must
come from a single computed fact. This trade-off (and the residual risk) is exactly why
the verifier reports its evidence rather than pretending to be an oracle.

**Fallback for flaky tool-calling:** after 2 consecutive malformed native tool calls the
loop switches to a structured-JSON envelope
(`{"thought", "action", "args"}` / `{"action": "final", "answer"}`) rendered from the same
canonical history — permitted explicitly by the brief, and forceable via
`LLM_FORCE_JSON_MODE=1`.

**Why no framework:** the loop's transparency — what gets retried, what gets verified,
what gets refused — *is* the deliverable. A framework would hide exactly the parts being
assessed, and hand-rolling keeps prompts small enough for free-tier limits. The trade-off
is that memory, streaming, and parallel tool calls are as simple as I made them, no freer.

## Sandbox (restricted execution)

Threat model: contain LLM mistakes and prompt-injected mischief — filesystem/network/
process access and runaway compute — not a determined human adversary. Defense in depth:

1. **AST gate** — node whitelist (expressions, assignments, comprehensions, lambdas);
   denies imports, loops (with a "vectorize instead" hint), def/class, try/with,
   underscore names/attributes (kills `__class__` chains), `getattr`/`eval`/`open`/…, and
   file-writing methods (`to_csv`, `to_pickle`, …).
2. **Stripped namespace** — a copy of `df`, whitelisted `pd`/`np` proxies (no `read_csv`,
   no `pd.eval`), minimal builtins.
3. **Warm worker subprocess** — killed on timeout (5s) and lazily respawned; survives
   infinite loops and memory blow-ups. Deliberately a plain subprocess speaking
   length-framed pickle over stdin/stdout rather than `multiprocessing` (whose spawn mode
   re-executes the parent's main module — it broke under stdin-launched parents) and
   rather than `signal.SIGALRM` (main-thread-only — it would silently stop working inside
   Streamlit's script thread).

## Design decisions & assumptions

1. Dataset found pristine beyond the two issues above — the null-audit is documented
   evidence, not an omission.
2. Impute-0-and-keep for the 11 never-billed customers (count fidelity for the agent).
3. `TotalCharges` excluded from model features (collinearity + what-if consistency);
   still available to EDA queries.
4. No class reweighting — calibration is part of the product surface.
5. Served model is the train-split fit; looking up a training-split customer by ID is
   in-sample. Stated openly; acceptable at this scope.
6. Questions about nonexistent columns (e.g. **region**, which the brief's own example
   mentions but the dataset lacks) are answered with an explicit absence statement plus
   the closest real columns — by design, not failure.
7. Hypothetical customers: unspecified fields default to training median/mode and are
   **always disclosed** in the answer; structurally impossible combinations (e.g.
   `InternetService="No"` with `OnlineSecurity="Yes"`) are corrected and reported.
8. Provider is env-configured OpenAI-compatible: Groq by default, OpenRouter is a
   3-variable swap. Groq retired `llama-3.3-70b-versatile` mid-2026; the default is now
   `openai/gpt-oss-120b`.
9. Verification tolerance: rounding-aware + 0.5% relative; simple derivations of computed
   facts count as verified (with the ≥3-significant-digit rule for pairwise derivations);
   anything else is stripped, never shipped.
10. The notebook is self-contained (Colab requirement) and mirrors `data.py`/`model.py`
    logic; the duplication is deliberate and the artifact schema is shared.
11. Dockerfile authored without local Docker (not installed on the dev machine) — build
    verified by inspection, honestly noted here.

## Tests

76 tests, all offline — the agent loop is proven with a scripted `FakeLLM` against the
*real* tools, sandbox, and verifier, so the graded behaviors don't depend on (or spend)
API quota: sandbox escapes rejected (`import os`, `().__class__.__mro__`,
`df.to_csv`, `getattr`, loops…) and timeouts respawn; planted hallucinated numbers trigger
revision and then degradation with nothing unverified shipped; cleaning invariants;
model bundle roundtrip; hypothetical-default disclosure; JSON-mode fallback switch;
multi-turn memory; honest no-answer on provider outage.

## Docker

```bash
docker build -t churn-analyst .
docker run --rm -p 8501:8501 --env-file .env churn-analyst
```

## AI tool use (disclosure)

Built with Claude Code (Anthropic) as a pair-programming agent: it implemented modules,
tests, and docs under my direction — scope, stack, and design decisions (model choice,
metric rationale, verifier design, what to cut) were reviewed and steered by me, and I can
walk through and defend any part of the code. The git history reflects the real build
order, including bugs found and fixed during live testing.

## Reflection (the honest half-page)

**Hardest part:** the numeric-provenance verifier. Extracting "numbers the model claims"
from free text sounds trivial and isn't — percents vs fractions, rounding, complements
("57% stay"), currency, thousands separators, customer IDs that look like numbers, list
ordinals. The genuinely hard lesson was the *false-accept* found in live testing: with
enough ledger facts, pairwise derivations will collide with almost any low-precision
figure, so I had to make derived matches earn trust with precision. Verification that's
too loose is theater; too strict and every answer degrades — tuning that boundary was the
most interesting engineering in the project.

**What I learned / had to figure out:** Groq's model lineup had rotated (the brief's
suggested Llama models are gone — handled by making the model an env var and hardening
config errors); `multiprocessing`'s spawn mode re-executes the parent main module (the
sandbox is a plain subprocess with framed pickle because of it); and pandas 3.0's string
dtype changes needed portable idioms so the notebook runs on Colab's pandas 2.x too.

**With more time:** a critic agent that re-derives each claimed figure independently
(the ledger makes this cheap); a small eval set with measured hallucination rate (the
verification reports already produce the raw material); charts in answers; temporal
validation is impossible with this snapshot dataset, but I'd say so more loudly in the UI.

## Time log (honest)

Roughly a focused day, AI-assisted end-to-end: ~1h brief analysis, dataset profiling and
planning · ~1.5h cleaning + notebook (EDA, model comparison, metric writing) · ~0.5h
production model module · ~2h sandbox + tools (including the multiprocessing→subprocess
rewrite) · ~2h verifier + agent loop + offline tests · ~1h live testing against Groq
(caught the verifier false-accept and the retired-model issue) · ~1h Streamlit, README,
Docker, deploy. Stopped at: no critic agent, no eval-set report, no React frontend —
listed as future work rather than half-built.
