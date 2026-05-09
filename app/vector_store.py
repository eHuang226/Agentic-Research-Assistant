"""ChromaDB collection per research session for RAG-backed synthesis."""

from __future__ import annotations

import chromadb
from chromadb.utils import embedding_functions


def get_client(persist_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(client: chromadb.PersistentClient, session_id: str):
    """One collection per session so runs do not collide."""
    name = f"research_{session_id.replace('-', '_')}"[:512]
    # Default embedding works offline; swap for OpenAI in production if desired.
    emb = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(name=name, embedding_function=emb)


def add_documents(
    collection,
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    if not texts:
        return
    collection.add(documents=texts, metadatas=metadatas, ids=ids)


def query_context(collection, query: str, n_results: int = 8) -> list[dict]:
    if collection.count() == 0:
        return []
    res = collection.query(query_texts=[query], n_results=min(n_results, max(1, collection.count())))
    out: list[dict] = []
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]
    dists = res.get("distances") or [[]]
    for doc, meta, dist in zip(docs[0], metas[0], dists[0], strict=False):
        out.append({"text": doc, "metadata": meta or {}, "distance": dist})
    return out
