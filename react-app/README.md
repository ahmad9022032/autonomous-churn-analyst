# ⚛️ ChurnSight React App — React frontend + FastAPI backend

> **Live:** https://churnsight-web.onrender.com/ *(free tier — wakes in ~1 min after idle)*

This folder contains the **React-based application**: a React 19 single-page app with
proper routed pages and reusable components, talking to a **FastAPI** backend over a
JSON/streaming API. It is a separate interface from the Streamlit app in
[`../streamlit-app/`](../streamlit-app/) — but both drive the **exact same brain**: the
backend imports the unchanged `churn_agent` package (agent loop, six tools, sandbox,
model, numeric-provenance verifier), so answers, verification behavior, and multi-turn
memory are identical. Zero business logic is duplicated in this folder — the backend is
pure transport, the frontend is pure presentation.

## Structure

```
react-app/
├── Dockerfile                # multi-stage: node build → slim python runtime (one container)
├── backend/                  # FastAPI — transport only
│   ├── main.py               # routes; POST /api/chat streams agent events as NDJSON;
│   │                         #   serves the built frontend with SPA fallback in production
│   ├── sessions.py           # per-session Agent registry (multi-turn memory)
│   ├── schemas.py            # pydantic request models
│   └── requirements.txt      # fastapi + uvicorn (the agent comes from `pip install -e .`)
└── frontend/                 # React 19 + Vite + react-router
    ├── vite.config.js        # dev proxy /api → :8000
    └── src/
        ├── api/client.js     # fetch helpers + NDJSON stream reader + session id
        ├── components/       # Nav · ChatMessage · EventTrace · VerificationBadge
        │                     #   · StatTile · RiskBar · ErrorBoundary
        ├── pages/            # ChatPage · DatasetPage · ModelPage · WhatIfPage
        ├── App.jsx           # router + shell
        └── styles.css        # design tokens + all styling
```

## Run locally

```bash
# 1. backend — from the REPO ROOT (needs .env with your Groq key)
.venv/bin/uvicorn --app-dir react-app backend.main:app --port 8000

# 2. frontend — dev server with hot reload
cd react-app/frontend
npm install
npm run dev          # → http://localhost:5173  (proxies /api to :8000)
```

Production mode (what Render runs): `npm run build`, then the FastAPI process on :8000
serves the built app itself — one container, one URL.

## Pages

| Page | What it does |
|------|--------------|
| **💬 Chat** | Streaming chat with the agent — the plan, tool calls, self-checks and the verification verdict render **live as they happen**; every answer carries its verification badge (`✅ 3/3 figures verified`) and call/latency stats; example prompts; conversation reset |
| **🗂️ Dataset** | Live overview from the API: stat tiles, the data-cleaning notes, per-column distributions as bars |
| **🎯 Customer Risk** | Model metrics, customer lookup → risk meter with percentile, top factors (▲ pushes churn up / ▼ down), account snapshot |
| **🧪 What-If Lab** | Pick a customer, change any features (dropdowns driven by the model's real value domains), see baseline vs projected risk and the delta — structural consistency enforced server-side |

## API

| Endpoint | Purpose |
|----------|---------|
| `GET /api/health` | liveness + configured model |
| `GET /api/overview` | dataset schema, counts, cleaning notes |
| `GET /api/metrics` | holdout metrics of the served artifact |
| `GET /api/schema` | feature domains + defaults (powers the What-If controls) |
| `GET /api/customers/{id}` | risk score, percentile, top factors, snapshot |
| `POST /api/whatif` | baseline vs projected risk under feature changes |
| `POST /api/hypothetical` | score a partial hypothetical profile |
| `POST /api/chat` | **NDJSON stream**: agent events live, then the final `{type: "result"}` with answer + verification |
| `POST /api/reset` | clear a session's conversation memory |

## Deploy (Render free tier — one container, one URL)

Defined by [`Dockerfile`](Dockerfile) and the repo-root [`render.yaml`](../render.yaml)
blueprint: **render.com** → *New +* → *Blueprint* → connect the repo → paste
`LLM_API_KEY` when prompted → *Apply*. Every push to `main` auto-redeploys.
