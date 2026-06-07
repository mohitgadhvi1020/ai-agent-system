"""FastAPI entry point for the AI Agent System."""
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.agent.memory import ConversationMemory
from app.agent.orchestrator import AgentOrchestrator
from app.agent.tools import get_tool_definitions
from app.config import settings
from app.models.schemas import (
    AgentRequest,
    AgentResponse,
    BatchProcessRequest,
    BatchProcessResponse,
    DocumentInput,
    HealthResponse,
    SessionInfo,
    SessionListResponse,
    ToolCall,
)

# Session store: session_id -> ConversationMemory
SESSIONS: Dict[str, ConversationMemory] = {}

STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] AI Agent System | provider={settings.LLM_PROVIDER} "
          f"| tools={len(settings.ENABLED_TOOLS)}")
    yield
    SESSIONS.clear()


app = FastAPI(title="AI Agent System", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_memory(session_id: str) -> ConversationMemory:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = ConversationMemory(settings.MAX_MEMORY_TURNS)
    return SESSIONS[session_id]


@app.get("/")
async def index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AI Agent System API. See /docs."}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        llm_provider=settings.LLM_PROVIDER,
        available_tools=[t["name"] for t in get_tool_definitions()],
    )


@app.post("/agent/run", response_model=AgentResponse)
async def agent_run(req: AgentRequest):
    session_id = req.session_id or str(uuid.uuid4())
    memory = _get_memory(session_id)
    orchestrator = AgentOrchestrator(memory=memory, max_steps=req.max_steps)

    if req.document:
        result = await orchestrator.process_document(req.document)
    else:
        result = await orchestrator.run(req.query)

    return AgentResponse(
        session_id=session_id,
        response=result["response"],
        tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
        steps_taken=result["steps_taken"],
        classification=result["classification"],
        extracted_entities=result["extracted_entities"],
        actions_taken=result["actions_taken"],
        trace=result["trace"],
    )


@app.post("/agent/process-document", response_model=AgentResponse)
async def process_document(doc: DocumentInput):
    session_id = doc.session_id or str(uuid.uuid4())
    memory = _get_memory(session_id)
    orchestrator = AgentOrchestrator(memory=memory)
    result = await orchestrator.process_document(doc.content, doc.filename)
    return AgentResponse(
        session_id=session_id,
        response=result["response"],
        tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
        steps_taken=result["steps_taken"],
        classification=result["classification"],
        extracted_entities=result["extracted_entities"],
        actions_taken=result["actions_taken"],
        trace=result["trace"],
    )


@app.post("/agent/batch", response_model=BatchProcessResponse)
async def agent_batch(req: BatchProcessRequest):
    batch_id = f"BATCH-{str(uuid.uuid4())[:6].upper()}"
    results, succeeded, failed = [], 0, 0
    for doc in req.documents:
        try:
            orchestrator = AgentOrchestrator(memory=ConversationMemory())
            r = await orchestrator.process_document(doc.content, doc.filename)
            results.append(
                {
                    "filename": doc.filename,
                    "success": True,
                    "classification": r["classification"],
                    "actions_taken": r["actions_taken"],
                    "steps_taken": r["steps_taken"],
                }
            )
            succeeded += 1
        except Exception as e:  # noqa: BLE001
            results.append({"filename": doc.filename, "success": False, "error": str(e)})
            failed += 1

    return BatchProcessResponse(
        batch_id=batch_id,
        total=len(req.documents),
        succeeded=succeeded,
        failed=failed,
        results=results,
    )


@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions():
    sessions = [
        SessionInfo(session_id=sid, turns=len(mem.turns), created_at=mem.created_at)
        for sid, mem in SESSIONS.items()
    ]
    return SessionListResponse(sessions=sessions, total=len(sessions))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found")
    del SESSIONS[session_id]
    return {"status": "deleted", "session_id": session_id,
            "deleted_at": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
