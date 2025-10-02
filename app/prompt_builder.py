import os
from typing import List
from .types import RetrievalHit

def _int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

def build_messages(system_prompt: str, question: str, hits: List[RetrievalHit]) -> list[dict]:
    # Limites por caracteres (aprox. tokens ≈ chars/4)
    per_chunk_chars = _int_env("PER_CHUNK_CHARS", 1500)       # corte por chunk
    max_context_chars = _int_env("MAX_CONTEXT_CHARS", 16000)  # orçamento total
    max_context_chunks = _int_env("MAX_CONTEXT_CHUNKS", 6)    # nº máx. de trechos

    context_blocks = []
    used = 0
    truncated_any = False

    for i, h in enumerate(hits[:max_context_chunks], start=1):
        src = h.chunk.metadata.get("source") or h.chunk.metadata.get("doc_id")
        txt = (h.chunk.text or "").strip()
        if len(txt) > per_chunk_chars:
            txt = txt[:per_chunk_chars] + "…"
            truncated_any = True

        block = f"[{i}] (fonte: {src})\n{txt}\n"
        if used + len(block) > max_context_chars:
            break
        context_blocks.append(block)
        used += len(block)

    context_text = "\n\n".join(context_blocks) if context_blocks else "—"
    note = "\n\n(Observação: contexto truncado para caber no limite.)" if truncated_any or used >= max_context_chars else ""

    user_prompt = f"""Pergunta: {question}

Você recebeu trechos recuperados do corpus (chamados de CONTEXTO). 
Responda objetivamente usando o CONTEXTO quando útil e cite os índices [1], [2]... das fontes usadas.
Se faltar contexto, explique o que está faltando ou peça para ingerir documentos.

CONTEXTO:
{context_text}{note}
"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
