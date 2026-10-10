
import pytest

from app.ingestion import split_into_chunks


def test_empty_text_returns_no_chunks():
    assert split_into_chunks("") == []


def test_short_text_returns_one_chunk():
    result = split_into_chunks("Python is useful.")

    assert result == ["Python is useful."]


def test_long_text_creates_multiple_chunks():
    text = "Python supports many applications. " * 100

    result = split_into_chunks(
        text,
        chunk_size=200,
        overlap=30,
    )

    assert len(result) > 1
    assert all(chunk.strip() for chunk in result)


def test_invalid_chunk_size_raises_error():
    with pytest.raises(ValueError):
        split_into_chunks("Some text", chunk_size=0)


def test_invalid_overlap_raises_error():
    with pytest.raises(ValueError):
        split_into_chunks("Some text", chunk_size=100, overlap=100)
