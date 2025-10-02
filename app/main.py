from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from typing import List, Optional
from collections import Counter
import os
from .pipeline import RAGPipeline
from .config import get_settings

app = FastAPI(title="RAG Modular — API")
_pipeline = RAGPipeline()
_settings = get_settings()

ALLOWED_EXTS = {".txt", ".md", ".pdf", ".docx"}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "collection": _settings.COLLECTION_NAME}

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    """
    Aceita apenas .txt, .md, .pdf, .docx.
    Retorna também os arquivos ignorados (skipped).
    """
    accepted_paths: List[str] = []
    skipped: List[dict] = []

    os.makedirs("uploads", exist_ok=True)

    for f in files:
        name = f.filename or ""
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTS:
            skipped.append({"filename": name, "reason": f"unsupported extension {ext}"})
            # ainda assim consome o stream para não deixar pendente
            await f.read()
            continue

        dest = os.path.join("uploads", name)
        with open(dest, "wb") as out:
            out.write(await f.read())
        accepted_paths.append(dest)

    if not accepted_paths:
        return {"ingested_docs": 0, "ingested_chunks": 0, "skipped": skipped}

    result = _pipeline.ingest_files(accepted_paths)
    result["skipped"] = skipped
    return result

@app.post("/query")
async def query(
    question: str = Form(...),
    system_prompt: Optional[str] = Form(None),
    top_k: int = Form(6),
    stream: bool = Form(False),
):
    if stream:
        gen = _pipeline.query(question=question, system_prompt=system_prompt, top_k=top_k, stream=True)
        return StreamingResponse(gen, media_type="text/plain; charset=utf-8")
    else:
        resp = _pipeline.query(question=question, system_prompt=system_prompt, top_k=top_k, stream=False)
        if hasattr(resp, "answer"):
            return {
                "answer": resp.answer,
                "sources": [
                    {"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "source": h.chunk.metadata.get("source")}
                    for h in resp.hits
                ],
            }
        return {"answer": str(resp), "sources": []}

@app.post("/clear")
async def clear():
    return _pipeline.clear()

@app.get("/corpus")
def corpus():
    """
    Lista fontes presentes no índice, com contagem de chunks por fonte.
    Útil para auditar se algum arquivo indevido entrou.
    """
    data = _pipeline.index.get_all()
    metas = data.get("metadatas") or []
    sources = [m.get("source", "desconhecido") for m in metas]
    c = Counter(sources)
    total_chunks = sum(c.values())
    unique_sources = [{"source": k, "chunks": v} for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]
    return {"total_chunks": total_chunks, "sources": unique_sources}

@app.get("/ui")
def ui():
    with open(os.path.join(os.path.dirname(__file__), "..", "ui", "index.html"), "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)
