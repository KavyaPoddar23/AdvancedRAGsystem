
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Reorder retrieved chunks using a cross-encoder relevance model."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not chunks:
            return []

        pairs = [
            (question, chunk["text"])
            for chunk in chunks
        ]

        scores = self.model.predict(pairs)

        ranked = []
        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["rerank_score"] = float(score)
            ranked.append(item)

        ranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return ranked[:top_k]
