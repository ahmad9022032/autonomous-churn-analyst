"""Offline proof of the graded loop behaviors: plan, act, verify, revise, degrade.

FakeLLM scripts the model; the tools, ledger, and verifier are all REAL —
so these tests exercise the actual dispatch/verification machinery without
an API key or rate-limit cost.
"""

import pytest

from churn_agent.agent import Agent, AgentEvent
from churn_agent.config import AgentConfig
from churn_agent.llm import FakeLLM, LLMResponse, ToolCall

CONFIG = AgentConfig(api_key="offline", max_revisions=2, max_tool_rounds=6)


def tool_call(name, args, call_id="c1"):
    return LLMResponse(
        content="PLAN: compute churn rate -> answer",
        tool_calls=[ToolCall(id=call_id, name=name, args=args)],
    )


def collect_events():
    events = []
    return events, events.append


def test_fabricated_number_triggers_revision_then_passes():
    fake = FakeLLM(
        [
            tool_call("run_python", {"code": "(df['Churn'] == 'Yes').mean()"}),
            LLMResponse(content="Churn rate is 26.5%, and revenue is $9,999,999."),
            LLMResponse(content="Churn rate is 26.5% across 7043 customers."),
        ]
    )
    # 7043 isn't in the ledger from that single tool call — wait, run_python on a
    # scalar returns one fact; make the clean draft cite only verified figures.
    fake.script[2] = LLMResponse(content="The churn rate is 26.5%.")
    agent = Agent(CONFIG, llm=fake)
    events, on_event = collect_events()

    result = agent.ask("what is the churn rate?", on_event=on_event)

    assert result.answer == "The churn rate is 26.5%."
    assert result.verification is not None and result.verification.ok
    assert result.llm_calls == 3
    kinds = [e.kind for e in events]
    assert "plan" in kinds and "revision" in kinds and "final" in kinds
    # the revision message must show the model its unverified figure and the facts
    revision_msg = fake.seen_messages[2][-1]["content"]
    assert "$9,999,999" in revision_msg and "Computed facts" in revision_msg


def test_persistent_fabrication_degrades_not_ships():
    bad = "Total revenue is $123,456 and churn costs $77,777 annually."
    fake = FakeLLM(
        [
            tool_call("run_python", {"code": "(df['Churn'] == 'Yes').mean()"}),
            LLMResponse(content=bad),
            LLMResponse(content=bad),
            LLMResponse(content=bad),
        ]
    )
    agent = Agent(CONFIG, llm=fake)
    result = agent.ask("what does churn cost us?", on_event=None)

    assert result.verification is not None and not result.verification.ok
    assert "$123,456" not in result.answer and "$77,777" not in result.answer
    from churn_agent.verify import FactLedger, verify_draft

    ledger = FactLedger()  # empty: degraded text must stand on its own numbers
    recheck = verify_draft(result.answer, ledger, "q")
    # every number left in the degraded answer must be a ledger render value;
    # with an empty ledger the only acceptable content is the facts-only template
    assert "could not verify" in result.answer or recheck.ok


def test_error_then_recovery_via_self_check():
    fake = FakeLLM(
        [
            tool_call("run_python", {"code": "df['Region'].value_counts()"}),  # KeyError
            tool_call("run_python", {"code": "(df['Churn'] == 'Yes').mean()"}, call_id="c2"),
            LLMResponse(content="About 26.5% of customers churn."),
        ]
    )
    agent = Agent(CONFIG, llm=fake)
    events, on_event = collect_events()
    result = agent.ask("churn by region?", on_event=on_event)

    assert result.verification.ok
    statuses = [s.status for s in result.steps]
    assert statuses == ["error", "ok"]
    assert any(e.kind == "self_check_retry" for e in events)
    # the error envelope (with hint) must have been fed back to the model
    tool_reply = fake.seen_messages[1][-1]["content"]
    assert "error" in tool_reply and "KeyError" in tool_reply


def test_tool_round_budget_exhaustion_yields_honest_partial():
    fake = FakeLLM(
        [tool_call("run_python", {"code": "(df['Churn'] == 'Yes').mean()"}, call_id=f"c{i}") for i in range(12)]
    )
    agent = Agent(CONFIG, llm=fake)
    result = agent.ask("loop forever", on_event=None)

    assert "budget" in result.answer
    # the honest partial shows real ledger facts, nothing fabricated
    assert "0.26537" in result.answer and "[run_python]" in result.answer


def test_malformed_tool_calls_switch_to_json_mode():
    fake = FakeLLM(
        [
            LLMResponse(content=None, tool_calls=[ToolCall("x1", "run_python", {}, parse_error=True)]),
            LLMResponse(content=None, tool_calls=[ToolCall("x2", "run_python", {}, parse_error=True)]),
            # now in JSON mode: content is the JSON envelope
            LLMResponse(content='{"thought": "compute", "action": "run_python", "args": {"code": "(df[\'Churn\'] == \'Yes\').mean()"}}'),
            LLMResponse(content='{"thought": "done", "action": "final", "answer": "The churn rate is 26.5%."}'),
        ]
    )
    agent = Agent(CONFIG, llm=fake)
    result = agent.ask("churn rate?", on_event=None)

    assert agent._json_mode is True
    assert result.answer == "The churn rate is 26.5%."
    assert result.verification.ok
    # JSON-mode render: system prompt must carry the tool catalogue
    json_mode_system = fake.seen_messages[2][0]["content"]
    assert "run_python" in json_mode_system and "final" in json_mode_system


def test_memory_carries_between_questions():
    fake = FakeLLM(
        [
            tool_call("run_python", {"code": "(df['Churn'] == 'Yes').mean()"}),
            LLMResponse(content="The churn rate is 26.5%."),
            tool_call("run_python", {"code": "df.groupby('Contract')['Churn'].apply(lambda s: (s == 'Yes').mean())"}, call_id="c9"),
            LLMResponse(content="Month-to-month is highest at 42.7%."),
        ]
    )
    agent = Agent(CONFIG, llm=fake)
    first = agent.ask("churn rate?", on_event=None)
    second = agent.ask("now break that down by contract", on_event=None)

    assert first.verification.ok and second.verification.ok
    # second episode's message history must contain the first (q, a) pair
    second_history = fake.seen_messages[2]
    assert any(m["content"] == "The churn rate is 26.5%." for m in second_history)


def test_llm_unavailable_yields_honest_no_answer():
    from churn_agent.llm import LLMUnavailable

    class DeadLLM:
        calls_made = 0

        def chat(self, messages, tools=None):
            raise LLMUnavailable("simulated outage")

    agent = Agent(CONFIG, llm=DeadLLM())
    result = agent.ask("anything", on_event=None)
    assert "could not reach" in result.answer
    assert "fabricated" in result.answer
