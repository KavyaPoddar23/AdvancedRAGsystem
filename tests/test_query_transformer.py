
import pytest

from app.query_transformer import QueryTransformer


def test_empty_question_is_rejected():
    transformer = QueryTransformer.__new__(QueryTransformer)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        transformer.transform("   ")


def test_invalid_json_is_rejected(monkeypatch):
    class FakeResponse:
        text = "not valid JSON"

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeResponse()

    class FakeClient:
        models = FakeModels()

    transformer = QueryTransformer.__new__(QueryTransformer)
    transformer.client = FakeClient()

    with pytest.raises(ValueError, match="invalid JSON"):
        transformer.transform("How does Python work?")
