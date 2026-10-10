
from collections import defaultdict

from rank_bm25 import BM25Okapi

from app.vector_store import VectorStore, load_chunks


def tokenize(text: str) -> list[str]:
    """Simple lowercase tokenizer for BM25."""
    return text.lower().split()


class HybridRetriever:
    """Combine semantic vector search and BM25 keyword search."""

    def __init__(self, vector_store: VectorStore, chunks: list[dict]):
        if not chunks:
            raise ValueError("Hybrid retrieval requires document chunks.")

        self.vector_store = vector_store
        self.chunks = chunks

        tokenized_documents = [
            tokenize(chunk["text"]) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_documents)

        # Map each chunk ID to its metadata.
        self.chunk_by_id = {
            chunk["chunk_id"]: chunk for chunk in chunks
        }

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Return chunks ranked by BM25 keyword relevance."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        scores = self.bm25.get_scores(tokenize(query))
        count = min(top_k, len(self.chunks))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:count]

        results = []

        for index in ranked_indices:
            result = dict(self.chunks[index])
            result["bm25_score"] = float(scores[index])
            results.append(result)

        return results

    @staticmethod
    def reciprocal_rank_fusion(
        ranked_lists: list[list[dict]],
        k: int = 60,
    ) -> list[dict]:
        """Fuse ranked lists using reciprocal rank fusion."""
        if k <= 0:
            raise ValueError("RRF k must be greater than zero.")

        fused_scores = defaultdict(float)
        chunk_lookup = {}

        for ranked_list in ranked_lists:
            for rank, chunk in enumerate(ranked_list, start=1):
                chunk_id = chunk["chunk_id"]
                fused_scores[chunk_id] += 1.0 / (k + rank)
                chunk_lookup[chunk_id] = chunk

        ordered_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )

        results = []

        for chunk_id in ordered_ids:
            result = dict(chunk_lookup[chunk_id])
            result["rrf_score"] = fused_scores[chunk_id]
            results.append(result)

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10,
    ) -> list[dict]:
        """Retrieve candidates from both methods and fuse their rankings."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be greater than zero.")

        vector_results = self.vector_store.search(
            query,
            top_k=min(candidate_k, len(self.chunks)),
        )
        keyword_results = self.keyword_search(
            query,
            top_k=min(candidate_k, len(self.chunks)),
        )

        fused_results = self.reciprocal_rank_fusion(
            [vector_results, keyword_results]
        )

        return fused_results[:top_k]


def main() -> None:
    """Run a sample hybrid retrieval query."""
    print("Loading document chunks...")
    chunks = load_chunks()

    print("Loading saved vector index...")
    vector_store = VectorStore()
    vector_store.load()

    # Ensure vector metadata and BM25 chunks refer to the same corpus.
    vector_ids = {chunk["chunk_id"] for chunk in vector_store.chunks}
    chunk_ids = {chunk["chunk_id"] for chunk in chunks}

    if vector_ids != chunk_ids:
        raise ValueError(
            "Vector index and processed chunks do not match. "
            "Rebuild the index with python -m app.vector_store."
        )

    retriever = HybridRetriever(vector_store, chunks)

    query = "How can SQL injection be prevented?"
    results = retriever.search(query, top_k=5)

    print(f"\nQuery: {query}")
    print("\nHybrid retrieval results:")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print(f"Source: {result['source']}")
        print(f"RRF score: {result['rrf_score']:.5f}")
        print(result["text"][:350])
        print("-" * 50)


if __name__ == "__main__":
    main()
