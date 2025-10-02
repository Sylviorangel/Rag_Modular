from typing import List
import os
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language,
)
from .types import Chunk

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

def split_chunks(doc_id: str, text: str, source: str) -> List[Chunk]:
    """
    Estratégias suportadas (via .env CHUNK_STRATEGY):
      - recursive_char (padrão): respeita separadores ["\n\n", "\n", " ", ""]
      - markdown: quebra por cabeçalhos (#, ##, ###) e depois refina com recursive
      - token: divide por tokens (tiktoken/cl100k_base)
      - code:<linguagem> ex.: code:python, code:js
    Demais parâmetros em .env: CHUNK_SIZE, CHUNK_OVERLAP
    """
    strategy = (os.getenv("CHUNK_STRATEGY", "recursive_char") or "recursive_char").lower()
    size = _env_int("CHUNK_SIZE", 1200)
    overlap = _env_int("CHUNK_OVERLAP", 150)

    parts: List[str] = []

    if strategy == "markdown":
        headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
        md = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
        md_docs = md.split_text(text)
        rec = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        for d in md_docs:
            parts.extend(rec.split_text(d.page_content))

    elif strategy == "token":
        tok = TokenTextSplitter(chunk_size=512, chunk_overlap=64, encoding_name="cl100k_base")
        parts = tok.split_text(text)

    elif strategy.startswith("code:"):
        # ex.: CHUNK_STRATEGY=code:python
        lang_name = strategy.split(":", 1)[1].upper()
        lang = getattr(Language, lang_name, Language.PYTHON)
        rec = RecursiveCharacterTextSplitter.from_language(language=lang, chunk_size=size, chunk_overlap=overlap)
        parts = rec.split_text(text)

    else:
        # recursive_char (padrão)
        rec = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        parts = rec.split_text(text)

    chunks: List[Chunk] = []
    for i, content in enumerate(parts):
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}::chunk_{i}",
                doc_id=doc_id,
                text=content,
                metadata={"source": source, "chunk_index": i},
            )
        )
    return chunks
