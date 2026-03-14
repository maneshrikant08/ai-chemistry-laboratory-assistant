import re
from typing import Iterable, List, Tuple
from langchain_core.documents import Document

from config import config

_UNIT_RE = re.compile(r"\bUNIT\s*[\-\u2013\u2014]\s*\d+\b.*", re.IGNORECASE)
_EXPERIMENT_RE = re.compile(r"^\s*EXPERIMENT\s*[:\-]?\s*(.+)$", re.IGNORECASE)
_SECTION_KEYWORDS = {
    "introduction",
    "theory",
    "procedure",
    "apparatus",
    "materials",
    "chemicals",
    "precautions",
    "safety",
    "observations",
    "result",
    "discussion",
    "questions",
}


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _detect_unit(line: str) -> str | None:
    match = _UNIT_RE.search(line)
    if match:
        return clean_text(match.group(0))
    return None


def _detect_section(line: str) -> str | None:
    norm = clean_text(line).lower()
    if len(norm) < 3:
        return None
    if norm in _SECTION_KEYWORDS:
        return norm.title()
    if line.isupper() and 3 <= len(norm) <= 80:
        return clean_text(line).title()
    return None


def _detect_experiment(line: str) -> str | None:
    match = _EXPERIMENT_RE.match(line.strip())
    if match:
        name = match.group(1).strip()
        return clean_text(name) if name else clean_text(line)
    if line.strip().lower().startswith("experiment"):
        return clean_text(line)
    return None


def _window_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_pdf_pages(
    pages: Iterable[Tuple[int, str]],
    doc_name: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[Document]:
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    documents: List[Document] = []
    current_unit = "UNKNOWN"
    current_section = "General"
    current_experiment = None
    buffer = []
    buffer_page = None

    def flush_buffer():
        nonlocal buffer, buffer_page
        if not buffer:
            return
        text = clean_text("\n".join(buffer))
        for chunk in _window_chunks(text, chunk_size, overlap):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_name": doc_name,
                        "unit_title": current_unit,
                        "section": current_section,
                        "experiment_name": current_experiment,
                        "page": buffer_page,
                    },
                )
            )
        buffer = []
        buffer_page = None

    for page_num, page_text in pages:
        if not page_text:
            continue
        lines = page_text.splitlines()
        for line in lines:
            unit = _detect_unit(line)
            if unit:
                flush_buffer()
                current_unit = unit
                current_section = "General"
                current_experiment = None
                continue

            experiment = _detect_experiment(line)
            if experiment:
                flush_buffer()
                current_experiment = experiment
                current_section = "General"
                continue

            section = _detect_section(line)
            if section:
                flush_buffer()
                current_section = section
                continue

            if buffer_page is None:
                buffer_page = page_num
            buffer.append(line)

        flush_buffer()

    return documents


def chunk_text(text: str, doc_name: str) -> List[Document]:
    cleaned = clean_text(text)
    chunks = _window_chunks(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    return [
        Document(
            page_content=chunk,
            metadata={
                "doc_name": doc_name,
                "unit_title": "UNKNOWN",
                "section": "General",
                "experiment_name": None,
                "page": None,
            },
        )
        for chunk in chunks
    ]
