import io
from pathlib import Path
from typing import List, Tuple

import fitz
from PIL import Image
import pytesseract
from docx import Document

from config import config


def _configure_tesseract():
    try:
        if config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    except Exception as exc:
        raise RuntimeError(f"Failed to configure Tesseract: {exc}") from exc


def load_pdf(path: Path) -> List[Tuple[int, str]]:
    try:
        doc = fitz.open(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{path.name}': {exc}") from exc

    pages: List[Tuple[int, str]] = []
    try:
        for i in range(doc.page_count):
            try:
                page = doc.load_page(i)
                text = page.get_text().strip()
                if not text:
                    _configure_tesseract()
                    pix = page.get_pixmap()
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    text = pytesseract.image_to_string(img)
                pages.append((i + 1, text))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to extract page {i + 1} from PDF '{path.name}': {exc}"
                ) from exc
        return pages
    finally:
        doc.close()


def load_docx(path: Path) -> str:
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:
        raise RuntimeError(f"Failed to read DOCX '{path.name}': {exc}") from exc


def load_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise RuntimeError(f"Failed to read TXT '{path.name}': {exc}") from exc


def load_image(path: Path) -> str:
    try:
        _configure_tesseract()
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as exc:
        raise RuntimeError(f"Failed to OCR image '{path.name}': {exc}") from exc
