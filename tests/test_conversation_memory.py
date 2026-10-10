
import pytest

from app.conversation_memory import ConversationMemory


def test_new_session_starts_empty():
    memory = ConversationMemory("test-session")

    assert memory.session_id == "test-session"
    assert memory.get_history() == []
    assert memory.get_state()["message_count"] == 0


def test_add_turn_stores_question_and_answer():
    memory = ConversationMemory("test-session")

    memory.add_turn("What is Python?", "A programming language.")

    history = memory.get_history()

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is Python?"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "A programming language."


def test_metadata_is_stored():
    memory = ConversationMemory("test-session")

    memory.add_turn(
        "What is RAG?",
        "Retrieval-augmented generation.",
        metadata={"sources": ["10_rag_architecture.txt"]},
    )

    assert memory.get_history()[1]["metadata"]["sources"] == [
        "10_rag_architecture.txt"
    ]


def test_get_context_returns_recent_turns():
    memory = ConversationMemory("test-session")

    memory.add_turn("Question one", "Answer one")
    memory.add_turn("Question two", "Answer two")

    context = memory.get_context(max_turns=1)

    assert len(context) == 2
    assert context[0]["content"] == "Question two"
    assert context[1]["content"] == "Answer two"


def test_empty_message_is_rejected():
    memory = ConversationMemory("test-session")

    with pytest.raises(ValueError, match="cannot be empty"):
        memory.add_message("user", "   ")


def test_invalid_role_is_rejected():
    memory = ConversationMemory("test-session")

    with pytest.raises(ValueError, match="Role must be"):
        memory.add_message("system", "Some content")


def test_empty_session_id_is_rejected():
    with pytest.raises(ValueError, match="Session ID cannot be empty"):
        ConversationMemory(" ")


def test_transcript_contains_both_roles():
    memory = ConversationMemory("test-session")
    memory.add_turn("What is Python?", "A programming language.")

    transcript = memory.format_transcript()

    assert "USER:" in transcript
    assert "ASSISTANT:" in transcript
    assert "What is Python?" in transcript
    assert "A programming language." in transcript


def test_clear_removes_messages():
    memory = ConversationMemory("test-session")
    memory.add_turn("Question", "Answer")

    memory.clear()

    assert memory.get_history() == []
    assert memory.get_state()["message_count"] == 0


def test_invalid_context_length_is_rejected():
    memory = ConversationMemory("test-session")

    with pytest.raises(ValueError, match="at least 1"):
        memory.get_context(max_turns=0)
