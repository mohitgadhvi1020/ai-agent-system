"""Pydantic request/response models."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    query: str = Field(..., description="Natural language task for the agent")
    session_id: Optional[str] = None
    document: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    max_steps: Optional[int] = None


class DocumentInput(BaseModel):
    content: str
    filename: str = "document.txt"
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None


class ToolCall(BaseModel):
    tool: str
    input: Dict[str, Any]
    output: Any
    step: int


class AgentResponse(BaseModel):
    session_id: str
    response: str
    tool_calls: List[ToolCall] = []
    steps_taken: int = 0
    classification: Optional[Dict[str, Any]] = None
    extracted_entities: Optional[Dict[str, Any]] = None
    actions_taken: List[str] = []
    trace: List[Dict[str, Any]] = []


class BatchProcessRequest(BaseModel):
    documents: List[DocumentInput]


class BatchProcessResponse(BaseModel):
    batch_id: str
    total: int
    succeeded: int
    failed: int
    results: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    available_tools: List[str]


class SessionInfo(BaseModel):
    session_id: str
    turns: int
    created_at: str


class SessionListResponse(BaseModel):
    sessions: List[SessionInfo]
    total: int
