"""Central paths and environment-driven configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = PROJECT_ROOT / "data" / "Customer-Churn.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

RANDOM_STATE = 42

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class AgentConfig:
    """Everything the agent loop needs; built from environment variables."""

    api_key: str = ""
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    force_json_mode: bool = False
    max_tool_rounds: int = 8
    max_llm_calls: int = 12
    max_revisions: int = 2
    sandbox_timeout_s: float = 5.0
    memory_turns: int = 6

    @classmethod
    def from_env(cls) -> "AgentConfig":
        return cls(
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", cls.base_url),
            model=os.getenv("LLM_MODEL", cls.model),
            force_json_mode=os.getenv("LLM_FORCE_JSON_MODE", "0") == "1",
        )
