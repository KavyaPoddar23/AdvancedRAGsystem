
from datetime import datetime, timezone
from typing import Any


class ConversationMemory:
    """Store and inspect a conversation for the current session."""

    def __init__(self, session_id: str = "default"):
        if not session_id.strip():
            raise ValueError("Session ID cannot be empty.")

        self.session_id = session_id
        self.messages: list[dict[str, Any]] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a user or assistant message to the session."""
        if role not in {"user", "assistant"}:
            raise ValueError("Role must be 'user' or 'assistant'.")

        if not content.strip():
            raise ValueError("Message content cannot be empty.")

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        self.messages.append(message)

    def add_turn(
        self,
        question: str,
        answer: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store one complete question-and-answer turn."""
        self.add_message("user", question)
        self.add_message("assistant", answer, metadata)

    def get_history(self) -> list[dict[str, Any]]:
        """Return a copy of the conversation history."""
        return [
            {
                **message,
                "metadata": dict(message["metadata"]),
            }
            for message in self.messages
        ]

    def get_context(self, max_turns: int = 5) -> list[dict[str, str]]:
        """Return recent messages for use as conversational context."""
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        recent_messages = self.messages[-(max_turns * 2):]

        return [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in recent_messages
        ]

    def get_state(self) -> dict[str, Any]:
        """Return an inspectable snapshot of the current session."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "message_count": len(self.messages),
            "messages": self.get_history(),
        }

    def format_transcript(self) -> str:
        """Format the conversation as readable text."""
        lines = [
            f"Session ID: {self.session_id}",
            f"Created at: {self.created_at}",
            "-" * 60,
        ]

        for message in self.messages:
            role = message["role"].upper()
            timestamp = message["timestamp"]
            content = message["content"]

            lines.append(f"[{timestamp}] {role}:")
            lines.append(content)
            lines.append("")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear messages without changing the session ID."""
        self.messages.clear()


def main():
    memory = ConversationMemory(session_id="phase-9-demo")

    memory.add_turn(
        "What is Python?",
        "Python is a high-level programming language.",
        metadata={"sources": ["01_python_basics.txt"]},
    )

    memory.add_turn(
        "What are its common data structures?",
        "Common Python data structures include lists, tuples, sets, and dictionaries.",
        metadata={"sources": ["02_python_data_structures.txt"]},
    )

    print("\nCONVERSATION TRANSCRIPT\n")
    print(memory.format_transcript())

    print("\nSESSION STATE\n")
    print(memory.get_state())

    print("\nRECENT CONTEXT\n")
    print(memory.get_context(max_turns=1))


if __name__ == "__main__":
    main()
