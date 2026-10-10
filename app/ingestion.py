
import json
from pathlib import Path

from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "chunks.jsonl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def read_txt_file(file_path: Path) -> list[dict]:
    """Read a text file and preserve its source filename."""
    text = file_path.read_text(encoding="utf-8-sig", errors="replace")

    return [{"text": text, "source": file_path.name, "page": None}]


def read_pdf_file(file_path: Path) -> list[dict]:
    """Extract text page by page from a PDF."""
    reader = PdfReader(str(file_path))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            records.append(
                {
                    "text": text,
                    "source": file_path.name,
                    "page": page_number,
                }
            )

    return records


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks while preserving word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1.")

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(start + 1, end - overlap)

        while next_start < len(text) and text[next_start] != " ":
            next_start += 1

        start = next_start

    return chunks


def load_documents() -> list[dict]:
    """Read all supported TXT and PDF files from the documents folder."""
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    file_paths = sorted(
        path
        for path in DOCUMENTS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".txt", ".pdf"}
    )

    if not file_paths:
        raise FileNotFoundError(
            f"No TXT or PDF documents found in {DOCUMENTS_DIR}"
        )

    records = []

    for file_path in file_paths:
        try:
            if file_path.suffix.lower() == ".pdf":
                file_records = read_pdf_file(file_path)
            else:
                file_records = read_txt_file(file_path)

            if not any(record["text"].strip() for record in file_records):
                print(
                    f"WARNING: No extractable text found in {file_path.name}"
                )
                continue

            records.extend(file_records)
            print(f"Read: {file_path.name}")

        except Exception as error:
            print(f"WARNING: Could not read {file_path.name}: {error}")

    if not records:
        raise ValueError("No readable document text was found.")

    return records


def create_chunks(records: list[dict]) -> list[dict]:
    """Split document text and retain source metadata for citations."""
    chunks = []

    for record in records:
        parts = split_into_chunks(record["text"])

        for index, part in enumerate(parts, start=1):
            chunks.append(
                {
                    "chunk_id": (
                        f"{record['source']}-"
                        f"{record['page'] or 'all'}-{index}"
                    ),
                    "text": part,
                    "source": record["source"],
                    "page": record["page"],
                    "chunk_index": index,
                }
            )

    return chunks


def save_chunks(chunks: list[dict]) -> None:
    """Save each chunk as one JSON object per line."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nSaved chunks to: {OUTPUT_FILE}")


def main() -> None:
    """Run the complete ingestion pipeline."""
    print("Starting document ingestion...\n")

    records = load_documents()
    chunks = create_chunks(records)

    if not chunks:
        raise ValueError("No chunks were created.")

    save_chunks(chunks)

    document_count = len({record["source"] for record in records})

    print(f"Documents processed: {document_count}")
    print(f"Text records read: {len(records)}")
    print(f"Chunks created: {len(chunks)}")
    print("\nIngestion completed successfully.")


if __name__ == "__main__":
    main()
