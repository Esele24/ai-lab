"""Turn an uploaded file into citable chunks.

The important idea: a chunk carries where it came from. An answer you cannot
trace back to a page is worth very little to someone checking a contract, so
page numbers survive extraction and chunking rather than being thrown away.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

SUPPORTED = (".pdf", ".docx", ".txt", ".md")


@dataclass
class Chunk:
    text: str
    source: str      # filename
    page: int | None # 1-based page number for PDFs, None otherwise
    index: int       # position within the document, for stable ordering

    @property
    def label(self) -> str:
        return f"{self.source} p.{self.page}" if self.page else self.source


def _read_pdf(data: bytes) -> list[tuple[int | None, str]]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    out: list[tuple[int | None, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # One unextractable page must not lose the other fifty.
            text = ""
        if text.strip():
            out.append((number, text))
    return out


def _read_docx(data: bytes) -> list[tuple[int | None, str]]:
    import docx

    document = docx.Document(io.BytesIO(data))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    # DOCX has no page concept until it is rendered, so page stays None rather
    # than being guessed. A made-up page number is worse than no page number.
    return [(None, "\n".join(paragraphs))] if paragraphs else []


def extract(filename: str, data: bytes) -> list[tuple[int | None, str]]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        if not data.startswith(b"%PDF-"):
            raise ValueError(f"{filename} is named .pdf but is not a PDF file.")
        return _read_pdf(data)
    if lower.endswith(".docx"):
        return _read_docx(data)
    if lower.endswith((".txt", ".md")):
        return [(None, data.decode("utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file type: {filename}. Supported: {', '.join(SUPPORTED)}")


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Split on sentence boundaries, packing up to `size` characters.

    Splitting mid-sentence is the classic RAG bug: the retrieved chunk reads as
    a fragment and the model fills the gap by inventing. The overlap exists so a
    fact sitting on a chunk boundary appears whole in at least one chunk.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= size:
            current = f"{current} {sentence}".strip()
            continue
        if current:
            chunks.append(current)
        # A single sentence longer than the window (tables, long clauses) is hard-cut.
        while len(sentence) > size:
            chunks.append(sentence[:size])
            sentence = sentence[size - overlap:]
        current = sentence
    if current:
        chunks.append(current)

    if overlap <= 0 or len(chunks) < 2:
        return chunks
    joined = [chunks[0]]
    for previous, chunk in zip(chunks, chunks[1:]):
        joined.append((previous[-overlap:] + " " + chunk).strip())
    return joined


def chunk_file(
    filename: str, data: bytes, *, size: int = 1200, overlap: int = 150
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page, text in extract(filename, data):
        for piece in _split(text, size, overlap):
            chunks.append(Chunk(text=piece, source=filename, page=page, index=len(chunks)))
    if not chunks:
        raise ValueError(
            f"No text could be extracted from {filename}. "
            "If it is a scanned PDF it needs OCR first -- this app does not do OCR."
        )
    return chunks
