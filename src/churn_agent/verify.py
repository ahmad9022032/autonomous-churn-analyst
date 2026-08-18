"""Numeric provenance: every figure in an answer must trace to a computed fact.

This is the anti-hallucination core the brief calls "the single most important
thing we're testing". Mechanism:

- FactLedger accumulates every numeric fact returned by tools this episode.
- extract_numbers() pulls the figures out of a draft answer (percents,
  currency, thousands separators; customer IDs and code spans excluded).
- Each figure must match a ledger fact under a small transform cascade —
  identity, x100/÷100 (percent forms), complement (100−x), absolute value,
  and pairwise derivations (ratio/difference/sum/share) of two facts.
  Matching is rounding-aware: "26.5%" verifies against 0.26537.
- Unmatched figures are returned so the agent can revise; if revision fails,
  degrade() strips the offending sentences (or falls back to a facts-only
  answer) — the degradation path itself can only emit ledger values.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

MAX_PAIRWISE_FACTS = 64

_CODE_SPAN = re.compile(r"```.*?```|`[^`]*`", re.DOTALL)
_CUSTOMER_ID = re.compile(r"\b\d{4}-[A-Z]{5}\b")
_ORDINAL = re.compile(r"(?m)^\s*(\d{1,2})[.)]\s")
_TOP_N = re.compile(r"\b(?:top|first|last|bottom)\s+(\d{1,2})\b", re.IGNORECASE)
_NUMBER = re.compile(
    r"(?<![\w.-])-?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.-])-?\$?\d+(?:\.\d+)?%?(?![\w.])"
)


@dataclass
class Fact:
    value: float
    label: str
    source: str
    step: int


@dataclass
class NumberMention:
    raw: str
    value: float
    decimals: int
    is_percent: bool
    span: tuple[int, int]


@dataclass
class VerificationReport:
    total: int = 0
    matched: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    whitelisted: int = 0
    attempts: int = 0
    redactions: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unmatched

    def summary(self) -> str:
        checked = len(self.matched) + len(self.unmatched)
        if checked == 0:
            return "no numeric claims to verify"
        state = "all verified" if self.ok else f"{len(self.unmatched)} UNVERIFIED"
        return f"{len(self.matched)}/{checked} figures verified against computed results ({state})"


class FactLedger:
    def __init__(self) -> None:
        self.facts: list[Fact] = []

    def add(self, facts: list[dict], source: str, step: int) -> None:
        for f in facts:
            try:
                self.facts.append(
                    Fact(value=float(f["value"]), label=str(f["label"]), source=source, step=step)
                )
            except (KeyError, TypeError, ValueError):
                continue

    def values(self) -> list[float]:
        return [f.value for f in self.facts]

    def render(self, limit: int = 40) -> str:
        """Compact fact list for revision prompts (deduped, most recent first)."""
        seen: set[tuple[str, float]] = set()
        lines: list[str] = []
        for f in reversed(self.facts):
            key = (f.label, round(f.value, 6))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {f.label} = {f.value:g}   [{f.source}]")
            if len(lines) >= limit:
                break
        return "\n".join(lines) if lines else "(no computed facts yet)"


def extract_numbers(text: str) -> list[NumberMention]:
    cleaned = _CODE_SPAN.sub(lambda m: " " * len(m.group()), text)
    cleaned = _CUSTOMER_ID.sub(lambda m: " " * len(m.group()), cleaned)
    skip_spans = [m.span(1) for m in _ORDINAL.finditer(cleaned)]
    skip_spans += [m.span(1) for m in _TOP_N.finditer(cleaned)]

    mentions: list[NumberMention] = []
    for m in _NUMBER.finditer(cleaned):
        if any(m.start() >= s and m.end() <= e for s, e in skip_spans):
            continue
        raw = m.group()
        stripped = raw.replace("$", "").replace(",", "")
        is_percent = stripped.endswith("%")
        stripped = stripped.rstrip("%")
        decimals = len(stripped.split(".")[1]) if "." in stripped else 0
        mentions.append(
            NumberMention(
                raw=raw,
                value=float(stripped),
                decimals=decimals,
                is_percent=is_percent,
                span=m.span(),
            )
        )
    return mentions


def _displays_as(mention: NumberMention, candidate: float) -> bool:
    """Does `candidate` round to exactly what the draft displayed? Plus a small
    relative tolerance for honest paraphrase (0.5%)."""
    if abs(round(candidate, mention.decimals) - mention.value) < 1e-9:
        return True
    return abs(mention.value - candidate) <= 0.005 * max(abs(candidate), 1e-9)


def _candidates(value: float) -> list[float]:
    out = [value, abs(value)]
    out += [value * 100, value / 100, abs(value) * 100]
    if 0.0 <= value <= 1.0:
        out += [1 - value, (1 - value) * 100]
    if 1.0 < value <= 100.0:
        out.append(100 - value)
    return out


def _matches_ledger(mention: NumberMention, values: list[float]) -> bool:
    for v in values:
        if any(_displays_as(mention, c) for c in _candidates(v)):
            return True
    # pairwise derived facts (ratio, difference, sum, share-of-total)
    pool = values[-MAX_PAIRWISE_FACTS:]
    for a, b in itertools.permutations(pool, 2):
        derived = [a - b, a + b]
        if b != 0:
            derived += [a / b, a / b * 100]
        if a + b != 0:
            derived += [a / (a + b), a / (a + b) * 100]
        if any(_displays_as(mention, d) for d in derived):
            return True
    return False


def verify_draft(draft: str, ledger: FactLedger, question: str) -> VerificationReport:
    report = VerificationReport()
    question_numbers = {m.value for m in extract_numbers(question)}
    values = ledger.values()

    for mention in extract_numbers(draft):
        report.total += 1
        if mention.value in question_numbers:
            report.whitelisted += 1
            continue
        if _matches_ledger(mention, values):
            report.matched.append(mention.raw)
        else:
            report.unmatched.append(mention.raw)
    return report


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n")


def degrade(draft: str, report: VerificationReport, ledger: FactLedger) -> str:
    """Remove unverifiable claims; never let them ship.

    Tier 1: drop the sentences containing unmatched figures, disclose the cut.
    Tier 2 (most of the answer was unverifiable): a facts-only answer built
    purely from the ledger, which by construction cannot hallucinate.
    """
    sentences = [s for s in _SENTENCE_SPLIT.split(draft) if s.strip()]
    bad = [s for s in sentences if any(u in s for u in report.unmatched)]
    keep = [s for s in sentences if s not in bad]
    report.redactions = report.unmatched[:]

    if sentences and len(bad) / len(sentences) > 0.5:
        lines = ledger.render(limit=15)
        return (
            "I could not verify enough of my draft answer against computed results, "
            "so here are only the directly computed facts:\n"
            f"{lines}\n\n"
            "Please rephrase the question if you need more."
        )
    note = (
        "\n\n*Note: some figures were removed from this answer because they could "
        "not be traced back to a computed result.*"
    )
    return " ".join(keep).strip() + note
