
import pytest

from app.hybrid_retriever import HybridRetriever


def make_chunk(chunk_id, text):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "source": f"{chunk_id}.txt",
        "page": None,
    }


def test_bm25_returns_results():
    chunks = [
        make_chunk(
            "sql",
            "SQL injection can be prevented with parameterized queries.",
        ),
        make_chunk(
            "python",
            "Python lists store multiple values.",
        ),
    ]

    # This test only exercises BM25, so no real vector model is needed.
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.chunks = chunks
    retriever.chunk_by_id = {
        chunk["chunk_id"]: chunk for chunk in chunks
    }

    from rank_bm25 import BM25Okapi

    retriever.bm25 = BM25Okapi(
        [chunk["text"].lower().split() for chunk in chunks]
    )

    results = retriever.keyword_search("SQL injection parameterized")

    assert results
    assert results[0]["chunk_id"] == "sql"


def test_rrf_combines_ranked_lists():
    first = make_chunk("a", "First chunk")
    second = make_chunk("b", "Second chunk")
    third = make_chunk("c", "Third chunk")

    results = HybridRetriever.reciprocal_rank_fusion(
        [
            [first, second],
            [first, third],
        ]
    )

    assert results[0]["chunk_id"] == "a"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]


def test_rrf_deduplicates_chunks():
    chunk = make_chunk("same", "Appears in both lists")

    results = HybridRetriever.reciprocal_rank_fusion(
        [[chunk], [chunk]]
    )

    assert len(results) == 1


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError, match="RRF k must be greater than zero"):
        HybridRetriever.reciprocal_rank_fusion([], k=0)


def test_keyword_search_rejects_empty_query():
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.chunks = [make_chunk("a", "Example text")]

    from rank_bm25 import BM25Okapi

    retriever.bm25 = BM25Okapi([["example", "text"]])

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.keyword_search("   ")
