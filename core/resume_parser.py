"""Resume file parser.

Handles PDF, DOCX, and TXT file uploads. Returns raw text that can be fed
to the LLM in Step 1 of the agent pipeline.
"""
from __future__ import annotations

import io
from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class UnsupportedFormatError(ValueError):
    """Raised when the uploaded file has an extension we don't support."""


class UnreadableFileError(ValueError):
    """Raised when the file content cannot be parsed (corrupt, empty, etc.)."""


class ResumeParser:
    """Extract plain text from uploaded resume files."""

    def parse(self, file_bytes: bytes, filename: str) -> str:
        """Route to the correct parser based on file extension.

        Args:
            file_bytes: Raw bytes from the uploaded file.
            filename: Original filename (used to detect extension).

        Returns:
            Plain text content, stripped of leading/trailing whitespace.

        Raises:
            UnsupportedFormatError: If the extension is not PDF/DOCX/TXT.
            UnreadableFileError: If parsing fails or the file is empty.
        """
        if not file_bytes:
            raise UnreadableFileError("Uploaded file is empty.")

        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported file format '{ext}'. "
                "Please upload PDF, DOCX, or TXT."
            )

        if ext == ".pdf":
            text = self.parse_pdf(file_bytes)
        elif ext == ".docx":
            text = self.parse_docx(file_bytes)
        else:
            text = self.parse_txt(file_bytes)

        text = text.strip()
        if not text:
            raise UnreadableFileError(
                "Could not extract any text from the file. "
                "Please try a different file or paste your resume as text."
            )
        return text

    def parse_pdf(self, file_bytes: bytes) -> str:
        """Extract text from a PDF using pdfplumber."""
        try:
            import pdfplumber
        except ImportError as e:
            raise UnreadableFileError(
                "pdfplumber is not installed. Run: pip install pdfplumber"
            ) from e

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n\n".join(pages)
        except Exception as e:
            raise UnreadableFileError(f"Could not read PDF: {e}") from e

    def parse_docx(self, file_bytes: bytes) -> str:
        """Extract text from a DOCX using python-docx."""
        try:
            from docx import Document
        except ImportError as e:
            raise UnreadableFileError(
                "python-docx is not installed. Run: pip install python-docx"
            ) from e

        try:
            doc = Document(io.BytesIO(file_bytes))
            paragraphs = [p.text for p in doc.paragraphs]
            return "\n".join(paragraphs)
        except Exception as e:
            raise UnreadableFileError(f"Could not read DOCX: {e}") from e

    def parse_txt(self, file_bytes: bytes) -> str:
        """Decode TXT file bytes as UTF-8 with a lenient fallback."""
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Latin-1 is a single-byte encoding and decodes any byte sequence.
            return file_bytes.decode("latin-1", errors="replace")
