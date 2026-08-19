"""The plan-act-check loop: tools, deterministic self-checks, numeric verification.

Hand-rolled on purpose (no framework): the loop's transparency — what gets
retried, what gets verified, what gets refused — is the deliverable. Every
LLM message is compact and every tool result truncated, because free-tier
rate limits make loop efficiency part of the design.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import AgentConfig
from .llm import LLMResponse, LLMUnavailable, ToolCall, parse_json_object
from .prompts import (
    JSON_MODE_INSTRUCTIONS,
    SYSTEM_PROMPT,
    budget_exhausted_prompt,
    revision_prompt,
)
from .tools import TOOL_SCHEMAS, TOOLS, dispatch
from .verify import FactLedger, VerificationReport, degrade, verify_draft

MAX_TOOL_RESULT_CHARS = 1500


@dataclass
class AgentEvent:
    kind: str  # plan|tool_call|tool_result|self_check_retry|draft|verify_verdict|revision|final|error
    payload: dict


@dataclass
class StepRecord:
    tool: str
    args: dict
    status: str
    summary: str


@dataclass
class AgentResult:
    answer: str
    steps: list[StepRecord] = field(default_factory=list)
    verification: VerificationReport | None = None
    llm_calls: int = 0
    elapsed_s: float = 0.0


def _compact(env: dict, self_check: str | None) -> str:
    data = env["data"]
    text = data if isinstance(data, str) else json.dumps(data, default=str)
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "... [truncated]"
    out = {"status": env["status"], "result": text}
    if env.get("hint"):
        out["hint"] = env["hint"]
    if self_check:
        out["self_check"] = self_check
    return json.dumps(out)


def _suspicious(env: dict) -> str | None:
    """Deterministic sanity rules over returned facts — no extra LLM calls."""
    for f in env.get("facts", []):
        label = f["label"].lower()
        value = f["value"]
        if any(k in label for k in ("rate", "risk", "share", "percentile")) and value > 100:
            return f"suspicious: {f['label']}={value} is outside any plausible rate scale"
        if (label.startswith(("n", "count", "row"))) and value < 0:
            return f"suspicious: negative count {f['label']}={value}"
    if env["status"] == "ok" and isinstance(env["data"], str) and "NaN" in env["data"]:
        return "result contains NaN — check the computation or filter"
    return None


class Agent:
    """`ask()` runs one plan-act-check episode; memory keeps (q, a) pairs."""

    def __init__(self, config: AgentConfig, llm: Any | None = None):
        self.config = config
        self._llm = llm
        self.memory: list[tuple[str, str]] = []
        self._json_mode = config.force_json_mode

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from .llm import LLMClient

            self._llm = LLMClient(self.config)
        return self._llm

    def reset(self) -> None:
        self.memory.clear()

    # ------------------------------------------------------------ chat modes
    def _render_json_mode(self, messages: list[dict]) -> list[dict]:
        """Re-render canonical (native-format) history for the JSON-envelope mode."""
        tool_lines = [
            f"- {name}: {t['schema']['description']} args={json.dumps(t['schema']['parameters']['properties'])}"
            for name, t in TOOLS.items()
        ]
        rendered = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + JSON_MODE_INSTRUCTIONS + "\n".join(tool_lines),
            }
        ]
        for m in messages[1:]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                calls = [
                    {"action": tc["function"]["name"], "args": tc["function"]["arguments"]}
                    for tc in m["tool_calls"]
                ]
                rendered.append({"role": "assistant", "content": json.dumps(calls[0])})
            elif m["role"] == "tool":
                rendered.append({"role": "user", "content": f"TOOL RESULT: {m['content']}"})
            else:
                rendered.append({"role": m["role"], "content": m["content"] or ""})
        return rendered

    def _chat(self, messages: list[dict]) -> LLMResponse:
        if not self._json_mode:
            return self.llm.chat(messages, TOOL_SCHEMAS)
        resp = self.llm.chat(self._render_json_mode(messages), None, force_json=True)
        obj = parse_json_object(resp.content or "")
        if obj is None:
            return LLMResponse(content=resp.content)
        if obj.get("action") in (None, "final"):
            return LLMResponse(content=str(obj.get("answer") or obj.get("thought") or ""))
        args = obj.get("args") or {}
        return LLMResponse(
            content=str(obj.get("thought") or ""),
            tool_calls=[ToolCall(id="json_0", name=str(obj["action"]), args=args if isinstance(args, dict) else {})],
        )

    # ------------------------------------------------------------ the loop
    def ask(self, question: str, on_event: Callable[[AgentEvent], None] | None = None) -> AgentResult:
        emit = on_event or (lambda e: None)
        started = time.monotonic()
        ledger = FactLedger()
        steps: list[StepRecord] = []
        calls_start = getattr(self.llm, "calls_made", 0)

        def result(answer: str, verification: VerificationReport | None) -> AgentResult:
            self.memory.append((question, answer))
            self.memory = self.memory[-self.config.memory_turns :]
            emit(AgentEvent("final", {"answer": answer}))
            return AgentResult(
                answer=answer,
                steps=steps,
                verification=verification,
                llm_calls=getattr(self.llm, "calls_made", 0) - calls_start,
                elapsed_s=round(time.monotonic() - started, 2),
            )

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for q, a in self.memory:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user", "content": question})

        plan_emitted = False
        consecutive_malformed = 0
        failures: Counter[str] = Counter()
        revisions = 0

        for round_no in range(self.config.max_tool_rounds + self.config.max_revisions + 1):
            if getattr(self.llm, "calls_made", 0) - calls_start >= self.config.max_llm_calls:
                return result(budget_exhausted_prompt(ledger.render(limit=15)), None)
            try:
                resp = self._chat(messages)
            except LLMUnavailable as exc:
                emit(AgentEvent("error", {"message": str(exc)}))
                return result(
                    f"I could not reach the language-model provider ({exc}). "
                    "No answer was fabricated; please retry in a moment.",
                    None,
                )

            if resp.content and not plan_emitted and "PLAN:" in resp.content:
                plan_line = resp.content.split("PLAN:", 1)[1].strip().splitlines()[0]
                emit(AgentEvent("plan", {"plan": plan_line}))
                plan_emitted = True

            if resp.tool_calls:
                if all(tc.parse_error for tc in resp.tool_calls):
                    consecutive_malformed += 1
                else:
                    consecutive_malformed = 0
                if consecutive_malformed >= 2 and not self._json_mode:
                    self._json_mode = True
                    emit(AgentEvent("self_check_retry", {"note": "switching to JSON tool fallback"}))

                messages.append(
                    {
                        "role": "assistant",
                        "content": resp.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                            }
                            for tc in resp.tool_calls
                        ],
                    }
                )
                for tc in resp.tool_calls:
                    emit(AgentEvent("tool_call", {"tool": tc.name, "args": tc.args}))
                    if tc.parse_error:
                        env = {
                            "status": "error",
                            "data": "tool arguments were not valid JSON",
                            "facts": [],
                            "hint": "emit arguments as a single valid JSON object",
                        }
                    else:
                        env = dispatch(tc.name, tc.args)

                    self_check = _suspicious(env)
                    if env["status"] != "ok":
                        failures[tc.name] += 1
                        if failures[tc.name] >= 3:
                            self_check = (
                                "this tool has now failed repeatedly — try a different "
                                "approach or answer honestly with what you have"
                            )
                    ledger.add(env.get("facts", []), source=tc.name, step=round_no)

                    summary = (env["data"] if isinstance(env["data"], str) else json.dumps(env["data"], default=str))[:120]
                    steps.append(StepRecord(tool=tc.name, args=tc.args, status=env["status"], summary=summary))
                    emit(AgentEvent("tool_result", {"tool": tc.name, "status": env["status"], "summary": summary}))
                    if env["status"] != "ok" or self_check:
                        emit(AgentEvent("self_check_retry", {"note": self_check or env.get("hint") or env["status"]}))

                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": _compact(env, self_check)}
                    )
                continue

            # ---- no tool calls: this is a draft answer
            draft = (resp.content or "").strip()
            if draft.startswith("PLAN:"):
                lines = draft.splitlines()
                draft = "\n".join(lines[1:]).strip() or lines[0]
            if not draft:
                # never ship an empty answer (seen live: reasoning models can
                # return empty content); nudge once per revision budget
                if revisions < self.config.max_revisions:
                    revisions += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": "Your reply was empty. Either call a tool "
                            "or state your final answer now.",
                        }
                    )
                    emit(AgentEvent("self_check_retry", {"note": "empty reply — nudging"}))
                    continue
                return result(budget_exhausted_prompt(ledger.render(limit=15)), None)
            if self._json_mode and not steps and revisions < self.config.max_revisions:
                # JSON-mode stall guard (seen live): the model narrates what it
                # needs instead of emitting the tool call — correct it once
                revisions += 1
                messages.append({"role": "assistant", "content": draft})
                messages.append(
                    {
                        "role": "user",
                        "content": "You have not called any tool yet. If you need "
                        'data, reply ONLY with the tool-call JSON now, e.g. '
                        '{"action": "get_data_overview", "args": {}}. If you truly '
                        "need nothing, restate your final answer.",
                    }
                )
                emit(AgentEvent("self_check_retry", {"note": "final before any tool call — nudging"}))
                continue
            emit(AgentEvent("draft", {"draft": draft}))

            report = verify_draft(draft, ledger, question)
            report.attempts = revisions + 1
            emit(AgentEvent("verify_verdict", {"summary": report.summary(), "ok": report.ok}))

            if report.ok:
                return result(draft, report)
            if revisions < self.config.max_revisions:
                revisions += 1
                messages.append({"role": "assistant", "content": draft})
                messages.append(
                    {"role": "user", "content": revision_prompt(report.unmatched, ledger.render())}
                )
                emit(AgentEvent("revision", {"unmatched": report.unmatched, "attempt": revisions}))
                continue
            return result(degrade(draft, report, ledger), report)

        return result(budget_exhausted_prompt(ledger.render(limit=15)), None)
