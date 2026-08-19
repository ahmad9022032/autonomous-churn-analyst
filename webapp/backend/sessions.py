"""Per-session Agent registry: same Agent class the Streamlit app and CLI use."""

from __future__ import annotations

import threading
from collections import OrderedDict

from churn_agent.agent import Agent
from churn_agent.config import AgentConfig

MAX_SESSIONS = 50

_lock = threading.Lock()
_agents: OrderedDict[str, Agent] = OrderedDict()


def get_agent(session_id: str) -> Agent:
    with _lock:
        if session_id not in _agents:
            _agents[session_id] = Agent(AgentConfig.from_env())
            while len(_agents) > MAX_SESSIONS:  # drop the oldest session
                _agents.popitem(last=False)
        _agents.move_to_end(session_id)
        return _agents[session_id]


def reset(session_id: str) -> None:
    with _lock:
        agent = _agents.get(session_id)
    if agent is not None:
        agent.reset()
