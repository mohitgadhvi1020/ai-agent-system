"""Sliding-window conversation memory with compaction."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    user: str
    assistant: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Optional[Dict[str, Any]] = None


class ConversationMemory:
    """Keeps the most recent turns verbatim; compacts older ones into a summary."""

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self.turns: List[Turn] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_turn(
        self, user: str, assistant: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.turns.append(Turn(user=user, assistant=assistant, metadata=metadata))
        if len(self.turns) > self.max_turns:
            self._compact()

    def _compact(self) -> None:
        """Compress the oldest 40% of turns into a single summary turn."""
        cut = max(1, int(len(self.turns) * 0.4))
        old, recent = self.turns[:cut], self.turns[cut:]
        topics = []
        for t in old:
            snippet = t.user.strip().split("\n")[0][:60]
            if snippet:
                topics.append(snippet)
        summary = "Earlier in the conversation, the user discussed: " + "; ".join(
            topics
        )
        summary_turn = Turn(
            user="[conversation summary]",
            assistant=summary,
            metadata={"compacted_from": len(old)},
        )
        self.turns = [summary_turn] + recent

    def to_messages(self) -> List[Dict[str, str]]:
        """Flatten into alternating user/assistant messages for the LLM."""
        messages: List[Dict[str, str]] = []
        for t in self.turns:
            messages.append({"role": "user", "content": t.user})
            messages.append({"role": "assistant", "content": t.assistant})
        return messages

    def get_context_window(self, last_n: int = 5) -> List[Turn]:
        return self.turns[-last_n:]
