
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "vectors.npy"
METADATA_FILE = EMBEDDINGS_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    """Generate embeddings and retrieve relevant document chunks."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)

    def build(self, chunks: list[dict]) -> None:
        """Generate and store one embedding per chunk."""
        if not chunks:
            raise ValueError("Cannot build a vector store without chunks.")

        texts = [chunk["text"] for chunk in chunks]

        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.chunks = chunks
        self.embeddings = np.asarray(vectors, dtype=np.float32)

    def save(self) -> None:
        """Save vectors and their corresponding chunk metadata."""
        if not self.chunks or self.embeddings.size == 0:
            raise ValueError("Build the vector store before saving it.")

        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

        np.save(EMBEDDINGS_FILE, self.embeddings)

        with METADATA_FILE.open("w", encoding="utf-8") as file:
            json.dump(self.chunks, file, ensure_ascii=False, indent=2)

        print(f"Saved vectors: {EMBEDDINGS_FILE}")
        print(f"Saved metadata: {METADATA_FILE}")

    def load(self) -> None:
        """Load previously saved vectors and metadata."""
        if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
            raise FileNotFoundError(
                "Vector index not found. Run the build command first."
            )

        self.embeddings = np.load(EMBEDDINGS_FILE)

        with METADATA_FILE.open("r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                "Vector count does not match the number of metadata records."
            )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top matching chunks, ordered by cosine similarity."""
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not self.chunks or self.embeddings.size == 0:
            raise ValueError("Load or build the vector store before searching.")

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        scores = self.embeddings @ query_vector
        result_count = min(top_k, len(self.chunks))

        best_indices = np.argsort(scores)[::-1][:result_count]

        results = []

        for index in best_indices:
            result = dict(self.chunks[int(index)])
            result["score"] = float(scores[index])
            results.append(result)

        return results


def load_chunks() -> list[dict]:
    """Read processed chunks from the JSONL file."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_FILE}. "
            "Run python -m app.ingestion first."
        )

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        raise ValueError("The chunks file is empty.")

    return chunks


def main() -> None:
    """Build and save the vector index."""
    print("Loading document chunks...")
    chunks = load_chunks()
    print(f"Chunks loaded: {len(chunks)}")

    print("\nLoading embedding model and generating vectors...")
    store = VectorStore()
    store.build(chunks)
    store.save()

    print(f"\nEmbedding dimensions: {store.embeddings.shape[1]}")
    print(f"Total vectors: {len(store.embeddings)}")
    print("\nVector index built successfully.")


if __name__ == "__main__":
    main()
