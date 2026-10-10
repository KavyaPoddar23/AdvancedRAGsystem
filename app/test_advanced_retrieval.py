
from app.vector_store import VectorStore, load_chunks
from app.hybrid_retriever import HybridRetriever
from app.query_transformer import QueryTransformer
from app.reranker import Reranker


def main():
    question = "What is SQL injection, and how can developers prevent it?"

    chunks = load_chunks()
    vector_store = VectorStore()
    vector_store.load()

    hybrid = HybridRetriever(vector_store, chunks)
    transformer = QueryTransformer()
    reranker = Reranker()

    queries = transformer.transform(question)

    print("Original question:", question)
    print("\nTransformed queries:")
    for query in queries:
        print("-", query)

    candidates_by_id = {}
    for query in queries:
        for chunk in hybrid.search(query, top_k=5, candidate_k=10):
            candidates_by_id[chunk["chunk_id"]] = chunk

    candidates = list(candidates_by_id.values())
    results = reranker.rerank(question, candidates, top_k=5)

    print("\nFinal re-ranked results:")
    for result in results:
        print(f"\nSource: {result['source']}")
        print(f"Re-ranking score: {result['rerank_score']:.3f}")
        print(result["text"][:400])
        print("-" * 50)


if __name__ == "__main__":
    main()
