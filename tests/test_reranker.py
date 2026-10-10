
import numpy as np
import pytest

from app.reranker import Reranker


class FakeCrossEncoder:
    def predict(self, pairs):
        # Higher scores represent greater relevance.
        return np.array([
            0.2 if "unrelated" in text.lower() else 0.9
            for _, text in pairs
        ])


def test_reranker_orders_relevant_chunks_first():
    reranker = Reranker.__new__(Reranker)
    reranker.model = FakeCrossEncoder()

    chunks = [
        {"chunk_id": "a", "text": "Unrelated topic"},
        {"chunk_id": "b", "text": "SQL injection prevention"},
    ]

    results = reranker.rerank("How to prevent SQL injection?", chunks)

    assert results[0]["chunk_id"] == "b"
    assert results[0]["rerank_score"] > results[1]["rerank_score"]


def test_reranker_respects_top_k():
    reranker = Reranker.__new__(Reranker)
    reranker.model = FakeCrossEncoder()

    chunks = [
        {"chunk_id": "a", "text": "SQL security"},
        {"chunk_id": "b", "text": "Unrelated topic"},
    ]

    assert len(reranker.rerank("SQL security", chunks, top_k=1)) == 1


def test_reranker_rejects_empty_question():
    reranker = Reranker.__new__(Reranker)
    reranker.model = FakeCrossEncoder()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        reranker.rerank(" ", [])


def test_reranker_returns_empty_for_no_chunks():
    reranker = Reranker.__new__(Reranker)
    reranker.model = FakeCrossEncoder()

    assert reranker.rerank("Python", []) == []
