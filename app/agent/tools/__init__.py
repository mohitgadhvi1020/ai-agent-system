"""Six production-style agent tools.

Each tool is an async function backed by pattern matching / heuristics so the
whole agent runs with zero API keys. In production these would call real
services; the interfaces and outputs are shaped to match that reality.
"""
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# In-memory stores (stand-ins for databases / external services)
# ---------------------------------------------------------------------------
_TASKS: List[Dict[str, Any]] = []
_NOTIFICATIONS: List[Dict[str, Any]] = []

_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "kb-1",
        "title": "Refund Policy",
        "content": (
            "Customers may request a full refund within 30 days of purchase. "
            "Refunds are processed to the original payment method within 5-7 "
            "business days. Subscription refunds are prorated."
        ),
        "tags": ["refund", "billing", "policy", "payment", "subscription"],
    },
    {
        "id": "kb-2",
        "title": "SLA Tiers",
        "content": (
            "Standard tier guarantees 99.5% uptime with 24h support response. "
            "Premium tier guarantees 99.9% uptime with 4h response. Enterprise "
            "tier guarantees 99.99% uptime with 1h response and a dedicated CSM."
        ),
        "tags": ["sla", "uptime", "support", "enterprise", "tiers"],
    },
    {
        "id": "kb-3",
        "title": "Onboarding Checklist",
        "content": (
            "New customer onboarding: create workspace, invite team members, "
            "connect data sources, configure integrations, run first workflow, "
            "schedule a success review at day 14."
        ),
        "tags": ["onboarding", "setup", "checklist", "workspace", "integration"],
    },
    {
        "id": "kb-4",
        "title": "API Rate Limits",
        "content": (
            "Free plan: 60 requests/minute. Pro plan: 600 requests/minute. "
            "Enterprise: custom limits. Exceeding limits returns HTTP 429 with "
            "a Retry-After header. Use exponential backoff."
        ),
        "tags": ["api", "rate limit", "429", "requests", "throttling"],
    },
    {
        "id": "kb-5",
        "title": "Data Retention",
        "content": (
            "Logs are retained for 90 days. Backups are kept for 1 year. "
            "Deleted records are purged after 30 days. Customers can request "
            "data export or deletion at any time per GDPR."
        ),
        "tags": ["data", "retention", "gdpr", "backup", "deletion", "logs"],
    },
]

# Category -> weighted keywords for document classification
_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "invoice": [
        "invoice", "amount due", "total", "subtotal", "tax", "payment",
        "bill to", "due date", "net 30", "balance",
    ],
    "contract": [
        "agreement", "party", "terms", "hereby", "shall", "obligations",
        "termination", "governing law", "signature", "effective date",
    ],
    "support_ticket": [
        "issue", "error", "bug", "ticket", "help", "not working", "problem",
        "urgent", "support", "broken",
    ],
    "report": [
        "report", "summary", "analysis", "quarter", "revenue", "growth",
        "metrics", "findings", "results", "performance",
    ],
    "email": [
        "dear", "hi", "hello", "regards", "best", "sincerely", "subject",
        "reply", "forwarded", "sent from",
    ],
}


# ---------------------------------------------------------------------------
# Tool 1: classify_document
# ---------------------------------------------------------------------------
async def classify_document(text: str, filename: str = "") -> Dict[str, Any]:
    """Score text against 5 categories using weighted keyword matching."""
    lowered = (text + " " + filename).lower()
    scores: Dict[str, float] = {}
    for category, keywords in _CATEGORY_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        scores[category] = hits / len(keywords)

    if not any(scores.values()):
        return {
            "category": "unknown",
            "confidence": 0.0,
            "all_scores": scores,
        }

    best = max(scores, key=scores.get)
    # Confidence: blend raw keyword density with separation from runner-up.
    ranked = sorted(scores.values(), reverse=True)
    margin = ranked[0] - (ranked[1] if len(ranked) > 1 else 0)
    confidence = round(min(1.0, scores[best] * 4 + margin), 2)
    return {
        "category": best,
        "confidence": confidence,
        "all_scores": {k: round(v, 2) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# Tool 2: extract_entities
# ---------------------------------------------------------------------------
_PATTERNS = {
    "emails": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "phones": r"(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
    "amounts": r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
    "urls": r"https?://[^\s]+",
    "dates": (
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+"
        r"\d{1,2},?\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    ),
}


async def extract_entities(
    text: str, entity_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Regex-based extraction of common entity types."""
    wanted = entity_types or list(_PATTERNS.keys()) + ["names"]
    out: Dict[str, List[str]] = {}

    for etype in wanted:
        if etype in _PATTERNS:
            matches = re.findall(_PATTERNS[etype], text)
            # De-dup, keep order
            seen, uniq = set(), []
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    uniq.append(m.strip())
            out[etype] = uniq

    if "names" in wanted:
        names = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        seen, uniq = set(), []
        for n in names:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        out["names"] = uniq[:10]

    return out


# ---------------------------------------------------------------------------
# Tool 3: summarize_text
# ---------------------------------------------------------------------------
_IMPORTANT_WORDS = {
    "revenue", "growth", "important", "critical", "increase", "decrease",
    "key", "result", "conclusion", "summary", "recommend", "risk", "profit",
}


async def summarize_text(text: str, max_sentences: int = 3) -> Dict[str, Any]:
    """Extractive summarization by position + keyword scoring."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return {"summary": "", "word_count": 0, "key_phrases": [], "compression_ratio": 0.0}

    n = len(sentences)
    scored = []
    for i, sent in enumerate(sentences):
        score = 0.0
        if i == 0:
            score += 2.0
        if i == n - 1:
            score += 1.0
        words = re.findall(r"\w+", sent.lower())
        score += sum(1 for w in words if w in _IMPORTANT_WORDS)
        score += min(len(words) / 20.0, 1.0)  # mild length preference
        scored.append((score, i, sent))

    top = sorted(scored, reverse=True)[:max_sentences]
    top_in_order = [s for _, _, s in sorted(top, key=lambda x: x[1])]
    summary = " ".join(top_in_order)

    key_phrases = sorted(
        {w for w in re.findall(r"\w+", text.lower()) if w in _IMPORTANT_WORDS}
    )
    orig_words = len(re.findall(r"\w+", text))
    sum_words = len(re.findall(r"\w+", summary))
    ratio = round(sum_words / orig_words, 2) if orig_words else 0.0

    return {
        "summary": summary,
        "word_count": sum_words,
        "key_phrases": key_phrases,
        "compression_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Tool 4: create_task
# ---------------------------------------------------------------------------
async def create_task(
    title: str,
    description: str = "",
    priority: str = "medium",
    assignee: str = "unassigned",
) -> Dict[str, Any]:
    """Create an in-memory task object with a generated ID."""
    task = {
        "id": f"TSK-{str(uuid.uuid4())[:4].upper()}",
        "title": title,
        "description": description,
        "priority": priority,
        "assignee": assignee,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _TASKS.append(task)
    return task


# ---------------------------------------------------------------------------
# Tool 5: send_notification
# ---------------------------------------------------------------------------
async def send_notification(
    channel: str, message: str, urgency: str = "normal"
) -> Dict[str, Any]:
    """Simulate sending a notification to Slack/email/webhook."""
    notif = {
        "id": f"NTF-{str(uuid.uuid4())[:6].upper()}",
        "channel": channel,
        "message": message,
        "urgency": urgency,
        "status": "delivered",
        "delivered_at": datetime.now(timezone.utc).isoformat(),
    }
    _NOTIFICATIONS.append(notif)
    return notif


# ---------------------------------------------------------------------------
# Tool 6: search_knowledge_base
# ---------------------------------------------------------------------------
async def search_knowledge_base(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Keyword-overlap search over the pre-loaded knowledge base."""
    q_words = set(re.findall(r"\w+", query.lower()))
    results = []
    for article in _KNOWLEDGE_BASE:
        haystack = (
            article["title"].lower()
            + " "
            + article["content"].lower()
            + " "
            + " ".join(article["tags"])
        )
        h_words = set(re.findall(r"\w+", haystack))
        overlap = q_words & h_words
        if not overlap:
            continue
        score = round(len(overlap) / max(len(q_words), 1), 2)
        results.append(
            {
                "id": article["id"],
                "title": article["title"],
                "content": article["content"],
                "relevance": score,
            }
        )

    results.sort(key=lambda r: r["relevance"], reverse=True)
    return {"query": query, "results": results[:top_k], "total_matches": len(results)}


# ---------------------------------------------------------------------------
# Registry + LLM-facing tool definitions
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    "classify_document": classify_document,
    "extract_entities": extract_entities,
    "summarize_text": summarize_text,
    "create_task": create_task,
    "send_notification": send_notification,
    "search_knowledge_base": search_knowledge_base,
}


def get_tool_definitions() -> List[Dict[str, Any]]:
    """JSON-Schema tool definitions used by the LLM for tool calling."""
    return [
        {
            "name": "classify_document",
            "description": "Classify a document into invoice, contract, support_ticket, report, or email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Document text"},
                    "filename": {"type": "string", "description": "Optional filename"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "extract_entities",
            "description": "Extract emails, phones, amounts, dates, URLs, and names from text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subset of entity types to extract",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "summarize_text",
            "description": "Produce an extractive summary of the given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "max_sentences": {"type": "integer", "default": 3},
                },
                "required": ["text"],
            },
        },
        {
            "name": "create_task",
            "description": "Create a task/ticket in the work tracker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                    "assignee": {"type": "string"},
                },
                "required": ["title"],
            },
        },
        {
            "name": "send_notification",
            "description": "Send a notification to a channel (slack, email, webhook).",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "message": {"type": "string"},
                    "urgency": {"type": "string", "enum": ["low", "normal", "high"]},
                },
                "required": ["channel", "message"],
            },
        },
        {
            "name": "search_knowledge_base",
            "description": "Search the internal knowledge base for relevant articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    ]
