# AI Agent System

A multi-step AI agent built on a **ReAct-style reasoning loop**. The agent
receives a task, reasons about which tools to use, executes them in sequence,
observes the results, and repeats until the task is complete or the step limit
is reached. Supports **Claude** and **GPT** backends — and runs fully without
any API keys in heuristic demo mode.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)

---

## Highlights

- **ReAct loop** — observe → think → act → repeat, capped at 10 steps.
- **6 production-style tools** — classification, entity extraction,
  summarization, task creation, notifications, knowledge-base search.
- **Dual LLM support** — Claude (`input_schema` tool format) or OpenAI
  (`function` calling), switchable via env var.
- **Zero-key demo mode** — a deterministic planner drives the loop using
  pattern matching, so the entire system is demoable with no API keys.
- **Session memory** — sliding-window conversation memory with automatic
  compaction of older turns.
- **Polished UI** — chat interface with a live execution trace at `/`.

---

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # defaults to LLM_PROVIDER=none (no key needed)
uvicorn app.main:app --reload
```

Open <http://localhost:8000> for the demo UI, or <http://localhost:8000/docs>
for the Swagger API.

### Docker

```bash
docker build -t ai-agent-system .
docker run -p 8000:8000 ai-agent-system
```

---

## Architecture

```
Request → ReAct Orchestrator ⇄ LLM (Claude/OpenAI/heuristic)
                  │
                  ├─ Tool Registry (6 tools)
                  ├─ Session Memory (sliding window + compaction)
                  └─ Response (answer + tool-call log + reasoning trace)
```

See `architecture.html` for a portfolio-ready diagram.

---

## Endpoints

| Method | Path | Description |
| ------ | ---- | ----------- |
| GET | `/` | Demo UI |
| GET | `/health` | Status, provider, available tools |
| POST | `/agent/run` | Run the agent on a natural-language task |
| POST | `/agent/process-document` | Classify → extract → route a document |
| POST | `/agent/batch` | Batch-process multiple documents |
| GET | `/sessions` | List active sessions |
| DELETE | `/sessions/{id}` | Delete a session's memory |

### Example

```bash
curl -s localhost:8000/agent/run \
  -H 'content-type: application/json' \
  -d '{"query": "What is our refund policy?"}' | python -m json.tool
```

---

## Tools

1. **classify_document** — scores text across 5 categories (invoice, contract,
   support_ticket, report, email).
2. **extract_entities** — emails, phones, amounts, dates, URLs, names.
3. **summarize_text** — extractive summary with key phrases.
4. **create_task** — creates a tracked task (`TSK-XXXX`).
5. **send_notification** — Slack / email / webhook (simulated).
6. **search_knowledge_base** — ranked search over 5 built-in articles.

---

## Configuration

All settings live in `.env` (see `.env.example`). To use a real LLM, set
`LLM_PROVIDER=claude` (or `openai`) and provide the matching API key.
