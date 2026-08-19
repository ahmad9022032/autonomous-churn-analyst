"""Streamlit chat UI over the churn agent.

Pure presentation: imports the same Agent the CLI uses and renders the same
AgentEvents — plan, tool calls, self-checks, verification — live in an
st.status block, so the user can watch the agent refuse to invent numbers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

# Belt and braces for hosts that run the app without installing the package
# (e.g. a bare `streamlit run` outside the venv): put src/ on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Streamlit Cloud injects dashboard secrets into st.secrets; bridge them into
# the environment so the provider-agnostic AgentConfig.from_env() sees them.
# The try/except keeps local runs (no secrets.toml at all) working.
try:
    for _key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "LLM_FORCE_JSON_MODE"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

from churn_agent.agent import Agent, AgentEvent
from churn_agent.config import METRICS_PATH, AgentConfig

st.set_page_config(page_title="Churn Analyst Agent", page_icon="📉")

EXAMPLES = [
    "Which customers are most likely to churn, and does that correlate with contract type?",
    "What is the churn risk for customer 7590-VHVEG and what drives it?",
    "A senior on a month-to-month fiber contract, tenure 2, paying $95/month — churn risk?",
    "What if customer 7590-VHVEG moved to a two-year contract?",
    "Average churn risk by payment method",
    "Does churn risk correlate with region?",
]

STATUS_ICONS = {"ok": "✅", "empty": "⚠️", "error": "❌"}


@st.cache_resource(show_spinner="Loading dataset, model and sandbox…")
def _warm_up() -> dict:
    from churn_agent.tools import warm_up

    warm_up()
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


def _get_agent() -> Agent:
    if "agent" not in st.session_state:
        st.session_state.agent = Agent(AgentConfig.from_env())
    return st.session_state.agent


def _render_events_into(container):
    def on_event(event: AgentEvent) -> None:
        p = event.payload
        if event.kind == "plan":
            container.write(f"**PLAN** → {p['plan']}")
        elif event.kind == "tool_call":
            args = json.dumps(p["args"])
            container.write(f"🔧 `{p['tool']}({args[:90] + '…' if len(args) > 90 else args})`")
        elif event.kind == "tool_result":
            container.write(f"{STATUS_ICONS.get(p['status'], '⚠️')} {p['status']}: `{p['summary'][:110]}`")
        elif event.kind == "self_check_retry":
            container.write(f"⚠️ self-check: {p['note']}")
        elif event.kind == "revision":
            container.write(f"🚫 verification rejected draft (unverified: {', '.join(p['unmatched'])}) — revising")
        elif event.kind == "error":
            container.write(f"❌ {p['message']}")

    return on_event


def _verification_caption(result) -> str | None:
    if result.verification is None:
        return None
    v = result.verification
    icon = "✅" if v.ok else "🚫"
    return f"{icon} {v.summary()} · {result.llm_calls} LLM calls · {result.elapsed_s}s"


config = AgentConfig.from_env()
_warm_up()

with st.sidebar:
    st.title("📉 Churn Analyst")
    st.caption(
        "An autonomous data-analyst agent over the 7,043-customer telco churn "
        "dataset. It plans, computes with real tools, self-checks, and every "
        "number in an answer is verified against an actually-computed result."
    )
    if st.button("↺ Reset conversation"):
        _get_agent().reset()
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Try one:")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = example

if not config.api_key:
    st.warning(
        "No LLM API key configured. Copy `.env.example` to `.env` and set "
        "`LLM_API_KEY` (free key at console.groq.com), then restart."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("caption"):
            st.caption(message["caption"])

prompt = st.chat_input(
    "Ask about the data, a customer's churn risk, or a what-if…",
    disabled=not config.api_key,
)
if not prompt and "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.status("Working…", expanded=True) as status:
                result = _get_agent().ask(prompt, on_event=_render_events_into(status))
                label = "Done"
                if result.verification is not None:
                    label = result.verification.summary()
                status.update(label=label, state="complete", expanded=False)
            st.markdown(result.answer)
            caption = _verification_caption(result)
            if caption:
                st.caption(caption)
            st.session_state.messages.append(
                {"role": "assistant", "content": result.answer, "caption": caption}
            )
        except Exception:
            friendly = (
                "Something went wrong while answering — nothing was fabricated. "
                "Please try rephrasing the question, or reset the conversation."
            )
            st.error(friendly)
            st.session_state.messages.append({"role": "assistant", "content": friendly})
