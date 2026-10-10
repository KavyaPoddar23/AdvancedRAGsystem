
import json

from app.answer_generator import AnswerGenerator
from app.conversation_memory import ConversationMemory
from app.evidence_grader import EvidenceGrader
from app.followup_resolver import FollowUpResolver
from app.hybrid_retriever import HybridRetriever
from app.query_transformer import QueryTransformer
from app.reranker import Reranker
from app.vector_store import VectorStore, load_chunks
from datetime import datetime, timezone
from pathlib import Path

class CorrectiveRAG:
    """Retrieve, grade, retry once, and generate grounded answers."""

    def __init__(self):
        chunks = load_chunks()

        vector_store = VectorStore()
        vector_store.load()

        vector_ids = {
            item["chunk_id"] for item in vector_store.chunks
        }
        chunk_ids = {
            item["chunk_id"] for item in chunks
        }

        if vector_ids != chunk_ids:
            raise ValueError(
                "Vector index and chunks do not match. "
                "Rebuild the index before running corrective RAG."
            )

        self.retriever = HybridRetriever(vector_store, chunks)
        self.transformer = QueryTransformer()
        self.reranker = Reranker()
        self.grader = EvidenceGrader()
        self.generator = AnswerGenerator()
        self.followup_resolver = FollowUpResolver()

    def retrieve(
        self,
        question: str,
        focus_reason: str | None = None,
        conversation_context: list[dict] | None = None,
    ) -> list[dict]:
        """Retrieve and rerank evidence for a question."""

        search_question = question

        if conversation_context:
            history_text = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in conversation_context
            )

            search_question = (
                "Use the conversation history to interpret the latest "
                "question. Retrieve evidence relevant to the latest "
                "question, not unrelated earlier topics.\n\n"
                f"Conversation history:\n{history_text}\n\n"
                f"Latest question: {question}"
            )

        if focus_reason:
            search_question += (
                "\nFocus the search on this missing evidence: "
                f"{focus_reason}"
            )

        queries = self.transformer.transform(search_question)
        all_queries = list(dict.fromkeys([search_question, *queries]))

        candidates = {}

        for query in all_queries:
            results = self.retriever.search(
                query,
                top_k=5,
                candidate_k=10,
            )

            for chunk in results:
                candidates[chunk["chunk_id"]] = chunk

        return self.reranker.rerank(
            question,
            list(candidates.values()),
            top_k=5,
        )

    @staticmethod
    def _service_error_result(
        error: Exception,
        attempts: list[dict],
    ) -> dict:
        """Report service failure separately from insufficient evidence."""

        return {
            "answer": (
                "I couldn't complete the request because an LLM service "
                "request failed. Check your API quota, model access, and "
                "network connection, then try again."
            ),
            "sources": [],
            "insufficient_evidence": False,
            "attempts": attempts,
            "retry_performed": len(attempts) > 1,
            "error": str(error),
        }

    def answer(
        self,
        question: str,
        conversation_context: list[dict] | None = None,
    ) -> dict:
        """Resolve follow-ups, retrieve evidence, grade it, and answer."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        try:
            retrieval_question = self.followup_resolver.resolve(
                question,
                conversation_context=conversation_context,
            )
        except RuntimeError as error:
            return self._service_error_result(error, [])

        print(f"\nResolved question: {retrieval_question}")

        attempts = []
        focus_reason = None

        for attempt_number in (1, 2):
            try:
                chunks = self.retrieve(
                    retrieval_question,
                    focus_reason=focus_reason,
                    conversation_context=None,
                )

                grade = self.grader.grade(retrieval_question, chunks)

            except RuntimeError as error:
                return self._service_error_result(error, attempts)

            attempts.append({
                "attempt": attempt_number,
                "retrieved_chunk_ids": [
                    chunk["chunk_id"] for chunk in chunks
                ],
                "evidence_sufficient": grade["sufficient"],
                "grading_reason": grade["reason"],
            })

            if grade["sufficient"]:
                try:
                    result = self.generator.generate_answer(
                        retrieval_question,
                        chunks,
                    )
                except RuntimeError as error:
                    return self._service_error_result(error, attempts)

                result["attempts"] = attempts
                result["retry_performed"] = attempt_number == 2
                result["resolved_question"] = retrieval_question
                return result

            if attempt_number == 1:
                focus_reason = grade["reason"]

        return {
            "answer": (
                "I don't have enough information in the provided documents "
                "to answer that reliably."
            ),
            "sources": [],
            "insufficient_evidence": True,
            "attempts": attempts,
            "retry_performed": True,
            "resolved_question": retrieval_question,
        }


def print_answer(result: dict) -> None:
    """Print the answer, sources, and retrieval details."""

    print("\nAnswer:\n")
    print(result["answer"])

    print(
        "\nInsufficient evidence:",
        result["insufficient_evidence"],
    )
    print("Retry performed:", result["retry_performed"])

    if result.get("error"):
        print("\nService error:")
        print(result["error"])

    print("\nRetrieval attempts:")

    for attempt in result.get("attempts", []):
        print(
            f"Attempt {attempt['attempt']}: "
            f"sufficient={attempt['evidence_sufficient']}; "
            f"reason={attempt['grading_reason']}"
        )

    print("\nSources:")
    sources = result.get("sources", [])

    if not sources:
        print("No sources available.")
    else:
        for source in sources:
            print(
                f"- [{source['source_id']}] {source['source']}"
            )


def save_session(memory: ConversationMemory) -> None:
    """Save the conversation transcript and full session state."""

    transcript_dir = Path("transcripts")
    transcript_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    transcript_path = transcript_dir / f"session_{timestamp}.txt"
    state_path = transcript_dir / f"session_{timestamp}.json"

    transcript_path.write_text(
        memory.format_transcript(),
        encoding="utf-8",
    )

    state_path.write_text(
        json.dumps(
            memory.get_state(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSession files saved:")
    print(f"Transcript: {transcript_path}")
    print(f"Session state: {state_path}")


def main():
    rag = CorrectiveRAG()
    memory = ConversationMemory(session_id="rag-session")

    print("\nAdvanced RAG — Interactive Session")
    print("Ask questions about your documents.")
    print("Commands: :state, :transcript, :quit")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break

        if not question:
            continue

        command = question.lower()

        if command in {":quit", ":exit"}:
            break

        if command == ":state":
            print("\nSESSION STATE\n")
            print(
                json.dumps(
                    memory.get_state(),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            continue

        if command == ":transcript":
            print("\nCONVERSATION TRANSCRIPT\n")
            print(memory.format_transcript())
            continue

        try:
            result = rag.answer(
                question,
                conversation_context=memory.get_context(max_turns=5),
            )
        except (RuntimeError, ValueError) as error:
            print(f"\nRequest failed: {error}")
            continue

        print_answer(result)

        memory.add_turn(
            question,
            result["answer"],
            metadata={
                "resolved_question": result.get("resolved_question"),
                "sources": result.get("sources", []),
                "insufficient_evidence": result.get(
                    "insufficient_evidence",
                    False,
                ),
                "retry_performed": result.get(
                    "retry_performed",
                    False,
                ),
                "attempts": result.get("attempts", []),
                "error": result.get("error"),
            },
        )

    print("\nFinal conversation transcript:\n")
    print(memory.format_transcript())

    save_session(memory)
    
    print(
        f"\nSession finished. "
        f"Total messages: {len(memory.get_history())}"
    )


if __name__ == "__main__":
    main()
