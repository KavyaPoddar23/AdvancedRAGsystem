from sympy import python


python
from app.evaluate import calculate_metrics


def test_metrics_with_perfect_results():
    results = [
        {
            "service_error": False,
            "source_hit": True,
            "keyword_coverage": 1.0,
            "abstention_correct": True,
        },
        {
            "service_error": False,
            "source_hit": True,
            "keyword_coverage": 0.5,
            "abstention_correct": True,
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics["total_cases"] == 2
    assert metrics["successful_cases"] == 2
    assert metrics["service_errors"] == 0
    assert metrics["source_hit_rate"] == 1.0
    assert metrics["average_keyword_coverage"] == 0.75
    assert metrics["abstention_accuracy"] == 1.0


def test_metrics_exclude_service_errors():
    results = [
        {
            "service_error": False,
            "source_hit": True,
            "keyword_coverage": 1.0,
            "abstention_correct": True,
        },
        {
            "service_error": True,
            "source_hit": None,
            "keyword_coverage": None,
            "abstention_correct": None,
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics["total_cases"] == 2
    assert metrics["successful_cases"] == 1
    assert metrics["service_errors"] == 1
    assert metrics["source_hit_rate"] == 1.0
    assert metrics["average_keyword_coverage"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0


def test_metrics_handle_empty_results():
    metrics = calculate_metrics([])

    assert metrics["total_cases"] == 0
    assert metrics["successful_cases"] == 0
    assert metrics["service_errors"] == 0
    assert metrics["source_hit_rate"] is None
    assert metrics["average_keyword_coverage"] is None
    assert metrics["abstention_accuracy"] is None


def test_metrics_handle_missing_expected_source_and_keywords():
    results = [
        {
            "service_error": False,
            "source_hit": None,
            "keyword_coverage": None,
            "abstention_correct": True,
        }
    ]

    metrics = calculate_metrics(results)

    assert metrics["source_hit_rate"] is None
    assert metrics["average_keyword_coverage"] is None
    assert metrics["abstention_accuracy"] == 1.0
