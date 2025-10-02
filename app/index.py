import chromadb
from chromadb import PersistentClient
from typing import List, Dict, Any
from .types import Chunk

class ChromaIndex:
    def __init__(self, path: str, collection_name: str):
        self.client: PersistentClient = chromadb.PersistentClient(path=path)
        # Não passamos embedding function para controlar embeddings manualmente.
        self.col = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def clear(self):
        self.client.delete_collection(self.col.name)
        self.col = self.client.get_or_create_collection(
            name=self.col.name, metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        assert len(chunks) == len(embeddings)
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [c.metadata | {"doc_id": c.doc_id} for c in chunks]
        # add() com embeddings pré-calculados
        self.col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)

    def get_all(self) -> Dict[str, Any]:
        # ⚠️ Não inclua "ids" em include; o Chroma sempre retorna ids por padrão.
        return self.col.get(include=["documents", "metadatas"])

    def query_by_vector(self, query_embedding: List[float], top_k: int = 6):
        # Para query, "distances" é permitido em include
        return self.col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings", "uris"],
        )
