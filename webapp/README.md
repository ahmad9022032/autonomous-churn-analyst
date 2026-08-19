# ChurnSight Web — React frontend + FastAPI backend

The optional Stage 2 "extra marks" build: a React app with proper pages, components
and API integration. It behaves **exactly** like the Streamlit app because the
backend is a thin wrapper importing the *same* `churn_agent` package — same agent
loop, same six tools, same sandbox, same LLM client, same numeric-provenance
verifier. Zero business logic is duplicated.

```
webapp/
├── backend/                  # FastAPI — the only "backend code" is transport
│   ├── main.py               # routes; POST /api/chat streams agent events as NDJSON
│   ├── sessions.py           # per-session Agent registry (multi-turn memory)
│   ├── schemas.py            # pydantic request models
│   └── requirements.txt      # fastapi + uvicorn (agent comes from `pip install -e .`)
└── frontend/                 # React 19 + Vite + react-router
    ├── vite.config.js        # dev proxy /api → :8000
    └── src/
        ├── api/client.js     # fetch helpers + NDJSON stream reader + session id
        ├── components/       # Nav · ChatMessage · EventTrace · VerificationBadge · StatTile · RiskBar
        ├── pages/            # ChatPage · DatasetPage · ModelPage · WhatIfPage
        ├── App.jsx           # router + shell
        └── styles.css        # design tokens (validated palette), all styling
```

## Run locally

```bash
# 1. backend (from the repo root; needs .env with your Groq key)
.venv/bin/uvicorn webapp.backend.main:app --port 8000

# 2. frontend
cd webapp/frontend
npm install
npm run dev          # → http://localhost:5173  (proxies /api to :8000)
```

## Pages

| Page | What it does |
|------|--------------|
| **Chat** | Streaming chat with the agent — the plan, tool calls, self-checks and the verification verdict render live as they happen; verification badge + call/latency stats on every answer; example prompts; conversation reset |
| **Dataset** | Live overview from the API: stat tiles, the data-issues notes, per-column distributions as bars |
| **Customer Risk** | Model metrics, customer lookup → risk meter with percentile, top factors (▲ pushes churn up / ▼ down), account snapshot |
| **What-If Lab** | Pick a customer, change any features (dropdowns driven by the model's real value domains), see baseline vs projected risk and the delta — structural consistency enforced server-side |

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | liveness + configured model |
| `GET /api/overview` | dataset schema, counts, cleaning notes |
| `GET /api/metrics` | holdout metrics of the served artifact |
| `GET /api/schema` | feature domains + defaults (powers What-If controls) |
| `GET /api/customers/{id}` | risk score, percentile, top factors, snapshot |
| `POST /api/whatif` | baseline vs projected risk under changes |
| `POST /api/hypothetical` | score a partial hypothetical profile |
| `POST /api/chat` | **NDJSON stream**: agent events, then the final `{type: "result"}` with answer + verification |
| `POST /api/reset` | clear a session's conversation memory |
