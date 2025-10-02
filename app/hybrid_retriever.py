from typing import List, Tuple
from rank_bm25 import BM25Okapi
import re
from .types import Chunk, RetrievalHit

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def build_bm25(corpus: List[str]) -> BM25Okapi:
    tokenized = [_tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized)

def reciprocal_rank_fusion(ranks: List[List[int]], k: int = 60) -> List[Tuple[int, float]]:
    scores = {}
    for rank in ranks:
        for r, idx in enumerate(rank):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1.0)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_retrieve(
    question: str,
    chroma_get_all: dict,
    vector_topk: List[Tuple[int, float]],
    vector_ids: List[str],
    top_k: int = 6,
) -> List[int]:
    docs: List[str] = chroma_get_all["documents"]
    bm25 = build_bm25(docs)
    bm_scores = bm25.get_scores(_tokenize(question))
    bm_rank = sorted(range(len(docs)), key=lambda i: bm_scores[i], reverse=True)
    vec_rank = [idx for idx, _ in vector_topk]
    fused = reciprocal_rank_fusion([bm_rank, vec_rank], k=60)
    top = [idx for idx, _ in fused[:top_k]]
    return top

def to_hits(global_docs: dict, selected_idxs: List[int], scores: List[float] | None = None) -> List[RetrievalHit]:
    ids = global_docs["ids"]
    docs = global_docs["documents"]
    metas = global_docs["metadatas"]
    hits: List[RetrievalHit] = []
    for j, i in enumerate(selected_idxs):
        chunk = Chunk(chunk_id=ids[i], doc_id=metas[i].get("doc_id",""), text=docs[i], metadata=metas[i])
        s = scores[j] if scores and j < len(scores) else 1.0
        hits.append(RetrievalHit(chunk=chunk, score=s))
    return hits
