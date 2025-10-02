from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class ParsedDoc:
    doc_id: str
    source: str
    text: str
    metadata: Dict[str, Any]

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any]

@dataclass
class RetrievalHit:
    chunk: Chunk
    score: float

@dataclass
class QueryResponse:
    answer: str
    hits: List[RetrievalHit]
