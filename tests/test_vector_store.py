
import numpy as np
import pytest

from app.vector_store import VectorStore


class FakeModel:
    """Small deterministic model for tests; no model download needed."""

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        vectors = {
            "Python lists and dictionaries store collections of values.": [
                1.0, 0.0, 0.0
            ],
            "SQL databases store and query structured data.": [
                0.0, 1.0, 0.0
            ],
            "Python collections": [1.0, 0.0, 0.0],
        }

        result = np.array(
            [vectors[text] for text in texts],
            dtype=np.float32,
        )

        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-12)

        return result


def make_store():
    store = VectorStore.__new__(VectorStore)
    store.model = FakeModel()
    store.chunks = [
        {
            "chunk_id": "python-1",
            "text": "Python lists and dictionaries store collections of values.",
            "source": "02_python_data_structures.txt",
            "page": None,
        },
        {
            "chunk_id": "sql-1",
            "text": "SQL databases store and query structured data.",
            "source": "06_sql_and_databases.txt",
            "page": None,
        },
    ]
    store.embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return store


def test_search_returns_most_relevant_chunk_first():
    store = make_store()

    results = store.search("Python collections", top_k=2)

    assert results[0]["chunk_id"] == "python-1"
    assert results[0]["score"] > results[1]["score"]


def test_search_respects_top_k():
    store = make_store()

    results = store.search("Python collections", top_k=1)

    assert len(results) == 1


def test_empty_query_raises_error():
    store = make_store()

    with pytest.raises(ValueError, match="Query cannot be empty"):
        store.search("   ")


def test_invalid_top_k_raises_error():
    store = make_store()

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        store.search("Python collections", top_k=0)
