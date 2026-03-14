from pathlib import Path
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from utils.document_loader import load_pdf
from utils.chunking import chunk_pdf_pages


def _index_exists(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def _collect_metadata_values(documents: List[Document]) -> tuple[list[str], list[str]]:
    unit_titles = sorted({d.metadata.get("unit_title", "UNKNOWN") for d in documents})
    experiments = sorted(
        {d.metadata.get("experiment_name") for d in documents if d.metadata.get("experiment_name")}
    )
    return unit_titles, experiments


def build_index_from_data(data_dir: Path, embeddings) -> Tuple[FAISS, List[str], List[str]]:
    try:
        documents: List[Document] = []
        for pdf_path in sorted(data_dir.glob("*.pdf")):
            pages = load_pdf(pdf_path)
            documents.extend(chunk_pdf_pages(pages, pdf_path.name))

        if not documents:
            raise RuntimeError(f"No PDF documents found in data directory: {data_dir}")

        store = FAISS.from_documents(documents, embeddings)
        unit_titles, experiments = _collect_metadata_values(documents)
        return store, unit_titles, experiments
    except Exception as exc:
        raise RuntimeError(f"Failed to build index from '{data_dir}': {exc}") from exc


def load_or_build_index(data_dir: Path, index_dir: Path, embeddings) -> Tuple[FAISS, List[str], List[str]]:
    try:
        index_dir.mkdir(parents=True, exist_ok=True)

        if _index_exists(index_dir):
            store = FAISS.load_local(
                index_dir, embeddings, allow_dangerous_deserialization=True
            )
            docs = list(store.docstore._dict.values())
            unit_titles, experiments = _collect_metadata_values(docs)
            return store, unit_titles, experiments

        store, unit_titles, experiments = build_index_from_data(data_dir, embeddings)
        store.save_local(index_dir)
        return store, unit_titles, experiments
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load or build index at '{index_dir}': {exc}"
        ) from exc


def build_index_from_documents(documents: List[Document], embeddings) -> FAISS:
    try:
        if not documents:
            raise RuntimeError("No documents were provided for indexing")
        return FAISS.from_documents(documents, embeddings)
    except Exception as exc:
        raise RuntimeError(f"Failed to build index from uploaded documents: {exc}") from exc


def extract_experiments_from_store(store: FAISS) -> List[str]:
    try:
        docs = list(store.docstore._dict.values())
        _, experiments = _collect_metadata_values(docs)
        return experiments
    except Exception as exc:
        raise RuntimeError(f"Failed to extract experiments from vector store: {exc}") from exc
