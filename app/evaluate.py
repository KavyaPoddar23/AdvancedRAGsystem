import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.corrective_rag import CorrectiveRAG


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "golden_questions.json"
DEFAULT_RESULTS = PROJECT_ROOT / "evaluation" / "results.json"


def evaluate_case(rag, case):
    question = case["question"]
    expected_sources = set(case.get("expected_sources", []))
    expected_keywords = case.get("expected_keywords", [])
    should_abstain = case.get("should_abstain", False)

    result = rag.answer(question)

    answer = result.get("answer", "")
    actual_sources = {
        source.get("source", "")
        for source in result.get("sources", [])
        if source.get("source")
    }

    matched_sources = sorted(expected_sources & actual_sources)

    answer_lower = answer.lower()
    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() in answer_lower
    ]

    service_error = bool(result.get("error"))

    if service_error:
        abstention_correct = None
        source_hit = None
        keyword_coverage = None
    else:
        abstention_correct = (
            result.get("insufficient_evidence", False) == should_abstain
        )

        source_hit = (
            bool(expected_sources & actual_sources)
            if expected_sources
            else None
        )

        keyword_coverage = (
            len(matched_keywords) / len(expected_keywords)
            if expected_keywords
            else None
        )

    return {
        "id": case["id"],
        "question": question,
        "answer": answer,
        "expected_sources": sorted(expected_sources),
        "actual_sources": sorted(actual_sources),
        "matched_sources": matched_sources,
        "source_hit": source_hit,
        "expected_keywords": expected_keywords,
        "matched_keywords": matched_keywords,
        "keyword_coverage": keyword_coverage,
        "should_abstain": should_abstain,
        "actual_abstention": result.get("insufficient_evidence"),
        "abstention_correct": abstention_correct,
        "service_error": service_error,
        "error": result.get("error"),
        "attempts": result.get("attempts", []),
    }


def calculate_metrics(results):
    successful = [
        item for item in results
        if not item["service_error"]
    ]

    source_results = [
        item["source_hit"]
        for item in successful
        if item["source_hit"] is not None
    ]

    keyword_results = [
        item["keyword_coverage"]
        for item in successful
        if item["keyword_coverage"] is not None
    ]

    abstention_results = [
        item["abstention_correct"]
        for item in successful
        if item["abstention_correct"] is not None
    ]

    def average(values):
        return round(sum(values) / len(values), 3) if values else None

    return {
        "total_cases": len(results),
        "successful_cases": len(successful),
        "service_errors": len(results) - len(successful),
        "source_hit_rate": average(source_results),
        "average_keyword_coverage": average(keyword_results),
        "abstention_accuracy": average(abstention_results),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the corrective RAG system using a golden dataset."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N evaluation cases.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the golden-question JSON file.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")

    if not args.dataset.is_file():
        parser.error(f"Dataset file not found: {args.dataset}")

    with args.dataset.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        parser.error("The evaluation dataset must be a non-empty JSON list.")

    if args.limit is not None:
        cases = cases[:args.limit]

    rag = CorrectiveRAG()
    results = []

    print(f"Running {len(cases)} evaluation case(s)...")

    for index, case in enumerate(cases, start=1):
        print(f"\n[{index}/{len(cases)}] {case['id']}: {case['question']}")

        try:
            result = evaluate_case(rag, case)
        except Exception as error:
            result = {
                "id": case.get("id", f"case_{index}"),
                "question": case.get("question", ""),
                "answer": "",
                "service_error": True,
                "error": f"{type(error).__name__}: {error}",
                "source_hit": None,
                "keyword_coverage": None,
                "abstention_correct": None,
            }

        results.append(result)

        if result["service_error"]:
            print(f"  SERVICE ERROR: {result.get('error')}")
        else:
            print(f"  Source hit: {result['source_hit']}")
            print(f"  Keyword coverage: {result['keyword_coverage']}")
            print(f"  Abstention correct: {result['abstention_correct']}")

        # Pace requests to reduce the chance of hitting Gemini's
        # per-minute request limit. This does not guarantee quota availability.
        if index < len(cases):
            print("\nWaiting 60 seconds before the next case...")
            time.sleep(60)

    metrics = calculate_metrics(results)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "metrics": metrics,
        "results": results,
    }

    DEFAULT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_RESULTS.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n--- Evaluation Summary ---")
    print(json.dumps(metrics, indent=2))
    print(f"\nDetailed results saved to: {DEFAULT_RESULTS}")


if __name__ == "__main__":
    main()
