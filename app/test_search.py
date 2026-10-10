
from app.vector_store import VectorStore


def main():
    store = VectorStore()
    store.load()

    query = "How can SQL injection be prevented?"
    results = store.search(query, top_k=3)

    for result in results:
        print(f"\nSource: {result['source']}")
        print(f"Score: {result['score']:.3f}")
        print(result["text"][:350])
        print("-" * 50)


if __name__ == "__main__":
    main()
