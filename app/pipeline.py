import os
import uuid
from typing import List, Tuple
from .types import ParsedDoc, Chunk, RetrievalHit, QueryResponse
from .config import get_settings
from . import parse_clean
from . import chunk as chunker            # splitter simples (fallback)
from . import chunk_lang                  # splitter do langchain (opcional)
from .embeddings import OpenAIEmbedder
from .index import ChromaIndex
from .hybrid_retriever import hybrid_retrieve, to_hits
from .prompt_builder import build_messages
from .generate import OpenAIChat
from .postprocess import trim_answer

class RAGPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.index = ChromaIndex(path=self.settings.CHROMA_DIR, collection_name=self.settings.COLLECTION_NAME)
        self.embedder = OpenAIEmbedder(
            api_key=self.settings.OPENAI_API_KEY,
            api_base=self.settings.OPENAI_API_BASE,
            model=self.settings.EMBEDDING_MODEL,
        )
        self.chat = OpenAIChat(
            api_key=self.settings.OPENAI_API_KEY,
            api_base=self.settings.OPENAI_API_BASE,
            model=self.settings.CHAT_MODEL,
        )

    # 1) INGESTÃO
    def ingest_files(self, paths: List[str]) -> dict:
        parsed_docs: List[ParsedDoc] = []
        all_chunks: List[Chunk] = []

        use_lang_splitter = str(os.getenv("USE_LANGCHAIN_SPLITTER", "0")).lower() in ("1", "true", "yes")

        for path in paths:
            text = parse_clean.parse_to_text(path)
            if not text:
                continue
            doc_id = str(uuid.uuid4())
            source = os.path.basename(path)
            parsed = ParsedDoc(
                doc_id=doc_id,
                source=source,
                text=text,
                metadata={"source": source, "path": os.path.abspath(path)},
            )
            parsed_docs.append(parsed)

            # ===> AQUI trocamos a chamada do chunker <===
            if use_lang_splitter:
                chunks = chunk_lang.split_chunks(
                    doc_id=doc_id,
                    text=parsed.text,
                    source=parsed.metadata["source"],
                )
            else:
                chunks = chunker.simple_char_chunk(
                    doc_id=doc_id,
                    text=parsed.text,
                    max_chars=int(os.getenv("CHUNK_SIZE", "1200")),
                    overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
                    base_metadata={"source": parsed.metadata["source"]},
                )
            # ===========================================
            all_chunks.extend(chunks)

        if not all_chunks:
            return {"ingested_docs": 0, "ingested_chunks": 0}

        embeddings = self.embedder.embed([c.text for c in all_chunks])
        self.index.upsert_chunks(all_chunks, embeddings)

        return {"ingested_docs": len(parsed_docs), "ingested_chunks": len(all_chunks)}

    # 2) CONSULTA
    def query(self, question: str, system_prompt: str | None = None, top_k: int = 6, stream: bool = False):
        sys_prompt = system_prompt or self.settings.DEFAULT_SYSTEM_PROMPT

        all_docs = self.index.get_all()
        # ids vêm por padrão do Chroma.get()
        if not all_docs.get("ids"):
            if stream:
                def gen():
                    yield "Não há documentos ingeridos ainda. Use a rota /ingest para adicionar arquivos."
                return gen()
            return QueryResponse(
                answer="Não há documentos ingeridos ainda. Use a rota /ingest para adicionar arquivos.",
                hits=[],
            )

        # Vetoriza pergunta
        q_vec = self.embedder.embed([question])[0]

        # Consulta vetorial (⚠️ não incluir 'ids' no include)
        vector_query = self.index.col.query(
            query_embeddings=[q_vec],
            n_results=min(10, len(all_docs["ids"])),
            include=["documents", "metadatas", "distances"],
        )

        # Mapear para índices globais
        global_ids = all_docs["ids"]      # retornado por padrão
        vec_ids = vector_query["ids"][0]  # retornado por padrão
        vec_idxs_scores: List[Tuple[int, float]] = []
        for i, cid in enumerate(vec_ids):
            try:
                idx_global = global_ids.index(cid)
            except ValueError:
                continue
            dist = vector_query["distances"][0][i]
            score = 1.0 / (1.0 + dist)
            vec_idxs_scores.append((idx_global, score))

        # Fusão (BM25 + vetorial)
        selected_idxs = hybrid_retrieve(
            question=question,
            chroma_get_all=all_docs,
            vector_topk=vec_idxs_scores,
            vector_ids=vec_ids,
            top_k=top_k,
        )

        hits: List[RetrievalHit] = to_hits(all_docs, selected_idxs)
        messages = build_messages(sys_prompt, question, hits)

        if stream:
            def generator():
                for token in self.chat.stream(messages):
                    yield token
            return generator()
        else:
            answer = self.chat.complete(messages)
            answer = trim_answer(answer)
            return QueryResponse(answer=answer, hits=hits)

    # 3) LIMPAR
    def clear(self):
        self.index.clear()
        return {"cleared": True}
