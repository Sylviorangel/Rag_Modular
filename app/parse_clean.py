import os
from typing import Optional
from pypdf import PdfReader
from docx import Document

# Apenas leitura de formatos textuais suportados
ALLOWED_EXTS = {".txt", ".md", ".pdf", ".docx"}

def read_txt_or_md(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def parse_to_text(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTS:
        # Bloqueia arquivos de código (.py, .js etc.) e quaisquer outros não suportados
        return None
    if ext in (".txt", ".md"):
        return read_txt_or_md(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    return None
