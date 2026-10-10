
import json

import pytest

from app.evidence_grader import EvidenceGrader


class FakeResponse:
    def __init__(self, result):
        self.text = json.dumps(result)


class FakeModels:
    def __init__(self, result):
        self.result = result
        self.last_request = None

    def generate_content(self, **kwargs):
        self.last_request = kwargs
        return FakeResponse(self.result)


def make_grader(result):
    grader = EvidenceGrader.__new__(EvidenceGrader)

    class FakeClient:
        models = FakeModels(result)

    grader.client = FakeClient()
    return grader


def test_empty_question_is_rejected():
    grader = make_grader({"sufficient": True, "reason": "Relevant"})

    with pytest.raises(ValueError, match="Question cannot be empty"):
        grader.grade(" ", [])


def test_no_chunks_are_insufficient():
    grader = make_grader({"sufficient": True, "reason": "Relevant"})

    result = grader.grade("What is RAG?", [])

    assert result["sufficient"] is False
    assert "No evidence" in result["reason"]
    assert grader.client.models.last_request is None


def test_valid_grade_is_returned():
    grader = make_grader({
        "sufficient": True,
        "reason": "The excerpt directly addresses the question.",
    })

    result = grader.grade(
        "How can SQL injection be prevented?",
        [{
            "source": "07_sql_injection.txt",
            "text": "Parameterized queries prevent SQL injection.",
        }],
    )

    assert result["sufficient"] is True
    assert "directly addresses" in result["reason"]


def test_invalid_json_is_rejected():
    grader = make_grader({"sufficient": True, "reason": "Valid"})

    class InvalidResponse:
        text = "not JSON"

    grader.client.models.generate_content = lambda **kwargs: InvalidResponse()

    with pytest.raises(ValueError, match="invalid JSON"):
        grader.grade(
            "What is Python?",
            [{"source": "python.txt", "text": "Python is a language."}],
        )
