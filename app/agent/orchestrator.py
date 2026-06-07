"""ReAct-style agent orchestrator.

Drives an observe -> think -> act -> repeat loop. The "think" step is backed by
Claude or OpenAI when an API key is configured, and by a deterministic heuristic
planner otherwise so the agent is fully demoable with no keys.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from app.agent.memory import ConversationMemory
from app.agent.tools import TOOL_REGISTRY, get_tool_definitions
from app.config import settings


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------
async def _call_claude(
    messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], model: str
) -> Dict[str, Any]:
    """Call Claude with native tool-calling. Returns {text, tool_call}."""
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    claude_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        }
        for t in tools
    ]
    resp = await client.messages.create(
        model=model, max_tokens=1024, messages=messages, tools=claude_tools
    )
    text, tool_call = "", None
    for block in resp.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            tool_call = {"name": block.name, "input": block.input}
    return {"text": text, "tool_call": tool_call}


async def _call_openai(
    messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], model: str
) -> Dict[str, Any]:
    """Call OpenAI with function calling. Returns {text, tool_call}."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    oai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]
    resp = await client.chat.completions.create(
        model=model, messages=messages, tools=oai_tools, tool_choice="auto"
    )
    msg = resp.choices[0].message
    tool_call = None
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        tool_call = {"name": tc.function.name, "input": json.loads(tc.function.arguments)}
    return {"text": msg.content or "", "tool_call": tool_call}


class AgentOrchestrator:
    """Runs the ReAct loop against the tool registry."""

    def __init__(
        self,
        memory: Optional[ConversationMemory] = None,
        max_steps: Optional[int] = None,
        provider: Optional[str] = None,
    ):
        self.memory = memory or ConversationMemory(settings.MAX_MEMORY_TURNS)
        self.max_steps = max_steps or settings.MAX_AGENT_STEPS
        self.provider = provider or settings.LLM_PROVIDER
        self.tools = get_tool_definitions()

    # --- public entry points ------------------------------------------------
    async def run(self, task: str) -> Dict[str, Any]:
        return await self._execute_loop(task)

    async def process_document(
        self, content: str, filename: str = "document.txt"
    ) -> Dict[str, Any]:
        task = (
            f"Process this document named '{filename}'. Classify it, extract key "
            f"entities, and take an appropriate action.\n\nDOCUMENT:\n{content}"
        )
        result = await self._execute_loop(task, document_text=content, filename=filename)
        return result

    # --- core loop ----------------------------------------------------------
    async def _execute_loop(
        self,
        task: str,
        document_text: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        messages = self.memory.to_messages()
        messages.append({"role": "user", "content": task})

        tool_calls: List[Dict[str, Any]] = []
        trace: List[Dict[str, Any]] = []
        final_text = ""
        step = 0

        while step < self.max_steps:
            step += 1
            trace.append({"type": "think", "step": step, "detail": "Selecting next action"})

            decision = await self._think(
                task, messages, tool_calls, document_text, filename
            )

            if decision["tool_call"] is None:
                final_text = decision["text"] or "Task complete."
                trace.append({"type": "result", "step": step, "detail": final_text})
                break

            name = decision["tool_call"]["name"]
            args = decision["tool_call"]["input"]
            trace.append(
                {"type": "act", "step": step, "detail": f"Calling {name}", "input": args}
            )

            output = await self._execute_tool(name, args)
            tool_calls.append(
                {"tool": name, "input": args, "output": output, "step": step}
            )
            trace.append(
                {"type": "observe", "step": step, "detail": f"{name} returned", "output": output}
            )

            # Feed the tool result back into the conversation for the LLM path.
            messages.append(
                {
                    "role": "assistant",
                    "content": f"[called {name} with {json.dumps(args)}]",
                }
            )
            messages.append(
                {"role": "user", "content": f"[{name} result]: {json.dumps(output)}"}
            )
        else:
            final_text = "Reached maximum reasoning steps."

        classification, entities, actions = self._track_outputs(tool_calls)
        self.memory.add_turn(task, final_text)

        return {
            "response": final_text,
            "tool_calls": tool_calls,
            "steps_taken": step,
            "trace": trace,
            "classification": classification,
            "extracted_entities": entities,
            "actions_taken": actions,
        }

    # --- "think" step -------------------------------------------------------
    async def _think(
        self,
        task: str,
        messages: List[Dict[str, Any]],
        prior_calls: List[Dict[str, Any]],
        document_text: Optional[str],
        filename: Optional[str],
    ) -> Dict[str, Any]:
        if self.provider == "claude" and settings.ANTHROPIC_API_KEY:
            return await _call_claude(messages, self.tools, settings.CLAUDE_MODEL)
        if self.provider == "openai" and settings.OPENAI_API_KEY:
            return await _call_openai(messages, self.tools, settings.OPENAI_MODEL)
        return self._heuristic_plan(task, prior_calls, document_text, filename)

    def _heuristic_plan(
        self,
        task: str,
        prior_calls: List[Dict[str, Any]],
        document_text: Optional[str],
        filename: Optional[str],
    ) -> Dict[str, Any]:
        """Deterministic planner used when no LLM key is configured.

        Builds a sensible plan for document tasks (classify -> extract -> act)
        and answers knowledge/summarize requests directly.
        """
        called = {c["tool"] for c in prior_calls}
        text = document_text if document_text is not None else task
        lowered = task.lower()

        # Document pipeline
        is_doc = document_text is not None or any(
            w in lowered for w in ["document", "process", "invoice", "ticket", "classify"]
        )

        if is_doc:
            if "classify_document" not in called:
                return self._call("classify_document", {"text": text, "filename": filename or ""})
            if "extract_entities" not in called:
                return self._call("extract_entities", {"text": text})
            if "create_task" not in called:
                category = self._last_output(prior_calls, "classify_document").get(
                    "category", "document"
                )
                return self._call(
                    "create_task",
                    {
                        "title": f"Review {category}",
                        "description": f"Auto-created from processing {filename or 'document'}",
                        "priority": "high" if category == "support_ticket" else "medium",
                    },
                )
            if "send_notification" not in called:
                task_obj = self._last_output(prior_calls, "create_task")
                return self._call(
                    "send_notification",
                    {
                        "channel": "slack",
                        "message": f"New item to review: {task_obj.get('id', 'TSK')}",
                        "urgency": "normal",
                    },
                )
            return self._finish(prior_calls)

        # Knowledge base questions
        if any(w in lowered for w in ["how", "what", "policy", "limit", "sla", "refund", "search", "find"]):
            if "search_knowledge_base" not in called:
                return self._call("search_knowledge_base", {"query": task})
            return self._finish(prior_calls)

        # Summarization
        if "summar" in lowered:
            if "summarize_text" not in called:
                return self._call("summarize_text", {"text": text})
            return self._finish(prior_calls)

        # Entity extraction
        if any(w in lowered for w in ["extract", "email", "phone", "entit"]):
            if "extract_entities" not in called:
                return self._call("extract_entities", {"text": text})
            return self._finish(prior_calls)

        # Fallback: classify then stop
        if "classify_document" not in called:
            return self._call("classify_document", {"text": text})
        return self._finish(prior_calls)

    # --- helpers ------------------------------------------------------------
    @staticmethod
    def _call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"text": "", "tool_call": {"name": name, "input": args}}

    def _finish(self, prior_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = self._summarize_run(prior_calls)
        return {"text": summary, "tool_call": None}

    @staticmethod
    def _last_output(calls: List[Dict[str, Any]], tool: str) -> Dict[str, Any]:
        for c in reversed(calls):
            if c["tool"] == tool and isinstance(c["output"], dict):
                return c["output"]
        return {}

    def _summarize_run(self, calls: List[Dict[str, Any]]) -> str:
        if not calls:
            return "No actions were necessary for this task."
        parts = []
        for c in calls:
            out = c["output"]
            if c["tool"] == "classify_document" and isinstance(out, dict):
                parts.append(
                    f"Classified as **{out.get('category')}** "
                    f"({int(out.get('confidence', 0) * 100)}% confidence)."
                )
            elif c["tool"] == "extract_entities" and isinstance(out, dict):
                found = ", ".join(f"{len(v)} {k}" for k, v in out.items() if v)
                parts.append(f"Extracted {found or 'no entities'}.")
            elif c["tool"] == "create_task" and isinstance(out, dict):
                parts.append(f"Created task {out.get('id')} ({out.get('priority')} priority).")
            elif c["tool"] == "send_notification" and isinstance(out, dict):
                parts.append(f"Sent {out.get('urgency')} notification to {out.get('channel')}.")
            elif c["tool"] == "search_knowledge_base" and isinstance(out, dict):
                top = out.get("results", [])
                if top:
                    parts.append(
                        f"Found {len(top)} relevant article(s); top match: "
                        f"\"{top[0]['title']}\" — {top[0]['content']}"
                    )
                else:
                    parts.append("No relevant knowledge base articles found.")
            elif c["tool"] == "summarize_text" and isinstance(out, dict):
                parts.append(f"Summary: {out.get('summary')}")
        return " ".join(parts)

    async def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return await tool(**args)
        except TypeError as e:
            return {"error": f"Invalid arguments for {name}: {e}"}
        except Exception as e:  # noqa: BLE001 - surface tool errors gracefully
            return {"error": f"{name} failed: {e}"}

    def _track_outputs(
        self, calls: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
        classification, entities, actions = None, None, []
        for c in calls:
            out = c["output"]
            if c["tool"] == "classify_document" and isinstance(out, dict):
                classification = out
            elif c["tool"] == "extract_entities" and isinstance(out, dict):
                entities = out
            elif c["tool"] == "create_task" and isinstance(out, dict):
                actions.append(f"Created task {out.get('id')}")
            elif c["tool"] == "send_notification" and isinstance(out, dict):
                actions.append(f"Sent notification {out.get('id')}")
        return classification, entities, actions
