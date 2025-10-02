from typing import List
from .types import Chunk

def simple_char_chunk(
    doc_id: str,
    text: str,
    max_chars: int = 1200,
    overlap: int = 150,
    base_metadata: dict | None = None,
) -> List[Chunk]:
    base_metadata = base_metadata or {}
    chunks: List[Chunk] = []
    i = 0
    n = 0
    while i < len(text):
        end = min(len(text), i + max_chars)
        chunk_text = text[i:end]
        chunk_id = f"{doc_id}::chunk_{n}"
        meta = dict(base_metadata)
        meta.update({"chunk_index": n})
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text, metadata=meta))
        n += 1
        i = end - overlap if end - overlap > i else end
    return chunks
