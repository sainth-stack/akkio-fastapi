import os
import logging
from typing import List, Dict, Optional, Any

import chromadb
from chromadb import PersistentClient
from openai import OpenAI


LOGGER_NAME = "uae_legislation_ingest"
logger = logging.getLogger(LOGGER_NAME)


CHROMA_DIR = os.getenv("CHROMA_DIR", os.path.join("vector_store", "uae_legislation"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "uae_legislation_ar_v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")


_client: Optional[chromadb.Client] = None
_collection = None
_openai_client: Optional[OpenAI] = None


def get_client() -> chromadb.Client:
    global _client
    if _client is None:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        _client = PersistentClient(path=CHROMA_DIR)
    return _client


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        names = {c.name for c in client.list_collections()}
        if COLLECTION_NAME in names:
            _collection = client.get_collection(COLLECTION_NAME)
        else:
            _collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return _collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    client = get_openai()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def upsert_chunks(chunks: List[Dict[str, Any]]) -> int:
    """Upsert chunks into Chroma. Each chunk dict must include: id, text, metadata."""
    if not chunks:
        return 0
    col = get_collection()
    documents = [c["text"] for c in chunks]
    embeddings = embed_texts(documents)
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    col.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(chunks)


def query_chunks(query: str, top_k: int = 8, where: Optional[Dict[str, Any]] = None):
    col = get_collection()
    emb = embed_texts([query])[0]
    res = col.query(
        query_embeddings=[emb],
        n_results=top_k,
        where=where if where else None,
        include=["metadatas", "documents", "distances"],
    )
    return res


__all__ = [
    "get_client",
    "get_collection",
    "upsert_chunks",
    "query_chunks",
    "embed_texts",
    "CHROMA_DIR",
    "COLLECTION_NAME",
]


