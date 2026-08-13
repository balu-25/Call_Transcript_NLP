"""
text_extractor.py
-------------------
Handles requirement 1: user uploads transcript via PDF or types it directly.
"""

import io
import pdfplumber


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts all text from an uploaded PDF file (in-memory, nothing written to disk)."""
    text_chunks = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)
    return "\n".join(text_chunks).strip()


def clean_transcript_text(raw_text: str) -> str:
    """Light normalization: strip excessive blank lines/whitespace."""
    if not raw_text:
        return ""
    lines = [line.rstrip() for line in raw_text.splitlines()]
    lines = [line for line in lines if line.strip() != ""]
    return "\n".join(lines)
