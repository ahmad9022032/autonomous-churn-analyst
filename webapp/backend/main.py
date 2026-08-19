"""ChurnSight API — a thin FastAPI wrapper around the exact same agent stack.

Zero business logic here: the agent loop, tools, sandbox, model and verifier
are imported unchanged from churn_agent (the packages the Streamlit app and
CLI use). POST /api/chat streams the live agent events (plan, tool calls,
self-checks, verification) as NDJSON lines, ending with the final result.

Run:  uvicorn webapp.backend.main:app --port 8000
"""

from __future__ import annotations

import json
import queue
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from churn_agent.agent import AgentEvent, AgentResult
from churn_agent.config import METRICS_PATH, AgentConfig
from churn_agent.data import schema_summary
from churn_agent.tools import predict_churn, predict_hypothetical, warm_up, what_if
from churn_agent.model import get_model

from . import sessions
from .schemas import ChatRequest, HypotheticalRequest, ResetRequest, WhatIfRequest

app = FastAPI(title="ChurnSight API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    warm_up()  # dataset, model, sandbox worker — before the first request


def _envelope_or_404(env: dict) -> dict:
    if env["status"] == "error":
        raise HTTPException(status_code=404, detail=env["data"])
    return {"status": env["status"], "data": env["data"], "hint": env.get("hint")}


# ------------------------------------------------------------------ info
@app.get("/api/health")
def health() -> dict:
    config = AgentConfig.from_env()
    return {"ok": True, "model": config.model, "has_key": bool(config.api_key)}


@app.get("/api/overview")
def overview() -> dict:
    return schema_summary()


@app.get("/api/metrics")
def metrics() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="metrics not trained yet")
    return json.loads(METRICS_PATH.read_text())


@app.get("/api/schema")
def model_schema() -> dict:
    """Feature domains + defaults — powers the What-If form controls."""
    model = get_model()
    return {
        "categorical": {c: model.b["allowed_values"][c] for c in model.b["categorical"]},
        "numeric": model.b["numeric"],
        "defaults": model.b["defaults"],
    }


# ------------------------------------------------------------------ model
@app.get("/api/customers/{customer_id}")
def customer(customer_id: str) -> dict:
    return _envelope_or_404(predict_churn(customer_id))


@app.post("/api/whatif")
def whatif(req: WhatIfRequest) -> dict:
    return _envelope_or_404(what_if(req.customer_id, req.changes))


@app.post("/api/hypothetical")
def hypothetical(req: HypotheticalRequest) -> dict:
    return _envelope_or_404(predict_hypothetical(req.attributes))


# ------------------------------------------------------------------ chat
def _serialize_result(result: AgentResult) -> dict:
    verification = None
    if result.verification is not None:
        v = result.verification
        verification = {
            "ok": v.ok,
            "summary": v.summary(),
            "matched": v.matched,
            "unmatched": v.unmatched,
            "redactions": v.redactions,
        }
    return {
        "type": "result",
        "answer": result.answer,
        "verification": verification,
        "llm_calls": result.llm_calls,
        "tool_steps": len(result.steps),
        "elapsed_s": result.elapsed_s,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """Stream agent events as NDJSON; the last line is the final result."""
    agent = sessions.get_agent(req.session_id)
    events: queue.Queue = queue.Queue()

    def on_event(event: AgentEvent) -> None:
        events.put({"type": event.kind, **event.payload})

    def run() -> None:
        try:
            result = agent.ask(req.question, on_event=on_event)
            events.put(_serialize_result(result))
        except Exception:
            events.put(
                {
                    "type": "result",
                    "answer": "Something went wrong while answering — nothing was "
                    "fabricated. Please try rephrasing the question.",
                    "verification": None,
                    "llm_calls": 0,
                    "tool_steps": 0,
                    "elapsed_s": 0,
                }
            )
        events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            item = events.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/reset")
def reset(req: ResetRequest) -> dict:
    sessions.reset(req.session_id)
    return {"ok": True}
