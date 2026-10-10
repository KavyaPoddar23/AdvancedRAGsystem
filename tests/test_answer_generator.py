
import json

import pytest

from app.answer_generator import AnswerGenerator, ABSTENTION_MESSAGE


class FakeResponse:
    def __init__(self, data):
        self.text = json.dumps(data)


class FakeModels:
    def __init__(self, data):
        self.data = data
        self.last_request = None

    def generate_content(self, **kwargs):
        self.last_request = kwargs
        return FakeResponse(self.data)


class FakeClient:
    def __init__(self, data):
        self.models = FakeModels(data)


def make_generator(data):
    generator = AnswerGenerator.__new__(AnswerGenerator)
    generator.client = FakeClient(data)
    generator.model_name = "test-model"
    return generator


def test_empty_question_is_rejected():
    generator = make_generator({})

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generator.generate_answer(" ", [])


def test_no_chunks_returns_abstention_without_api_call():
    generator = make_generator({})

    result = generator.generate_answer("What is RAG?", [])

    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["sources"] == []
    assert result["insufficient_evidence"] is True
    assert generator.client.models.last_request is None


def test_grounded_answer_returns_valid_source():
    generator = make_generator({
        "answer": "Use parameterized queries to prevent SQL injection [S1].",
        "cited_sources": ["S1"],
        "insufficient_evidence": False,
    })

    chunks = [{
        "chunk_id": "sql-1",
        "text": "Parameterized queries help prevent SQL injection.",
        "source": "07_sql_injection.txt",
        "page": None,
    }]

    result = generator.generate_answer(
        "How can SQL injection be prevented?",
        chunks,
    )

    assert result["insufficient_evidence"] is False
    assert "[S1]" in result["answer"]
    assert result["sources"][0]["source"] == "07_sql_injection.txt"


def test_insufficient_evidence_returns_abstention():
    generator = make_generator({
        "answer": "The answer is unavailable.",
        "cited_sources": [],
        "insufficient_evidence": True,
    })

    chunks = [{
        "chunk_id": "python-1",
        "text": "Python supports lists and dictionaries.",
        "source": "02_python_data_structures.txt",
        "page": None,
    }]

    result = generator.generate_answer("What is quantum computing?", chunks)

    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["sources"] == []
    assert result["insufficient_evidence"] is True


def test_unknown_citation_causes_abstention():
    generator = make_generator({
        "answer": "The answer is supported [S9].",
        "cited_sources": ["S9"],
        "insufficient_evidence": False,
    })

    chunks = [{
        "chunk_id": "python-1",
        "text": "Python supports lists and dictionaries.",
        "source": "02_python_data_structures.txt",
        "page": None,
    }]

    result = generator.generate_answer("What is Python?", chunks)

    assert result["answer"] == ABSTENTION_MESSAGE
    assert result["sources"] == []
    assert result["insufficient_evidence"] is True


def test_invalid_json_is_rejected():
    class InvalidResponse:
        text = "not valid JSON"

    class InvalidModels:
        def generate_content(self, **kwargs):
            return InvalidResponse()

    generator = AnswerGenerator.__new__(AnswerGenerator)

    class Client:
        models = InvalidModels()

    generator.client = Client()
    generator.model_name = "test-model"

    with pytest.raises(ValueError, match="invalid JSON"):
        generator.generate_answer(
            "What is Python?",
            [{
                "chunk_id": "python-1",
                "text": "Python is a programming language.",
                "source": "01_python_basics.txt",
                "page": None,
            }],
        )
