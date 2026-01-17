# rag_store.py
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


STORE_DIR = Path("src/rag_store")
INDEX_PATH = STORE_DIR / "pokemon.index"
DOCS_PATH = STORE_DIR / "documents.jsonl"


# ---------------- Embedding model ----------------
@st.cache_resource
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ---------------- Load FAISS + docs ----------------
@st.cache_resource
def load_faiss_index(index_path: Path = INDEX_PATH):
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(str(index_path))


@st.cache_data
def load_docs(docs_path: Path = DOCS_PATH) -> List[Dict[str, Any]]:
    if not docs_path.exists():
        raise FileNotFoundError(f"documents.jsonl not found: {docs_path}")
    docs = []
    for line in docs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            docs.append(json.loads(line))
    return docs


# ---------------- Core query ----------------
def query_rag_store(
    query: str,
    k: int = 8,
    index_path: Path = INDEX_PATH,
    docs_path: Path = DOCS_PATH,
    min_score: Optional[float] = None,
) -> List[Tuple[float, Dict[str, Any]]]:
    embedder = get_embedder()
    index = load_faiss_index(index_path)
    docs = load_docs(docs_path)

    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, k)

    results: List[Tuple[float, Dict[str, Any]]] = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(docs):
            continue
        s = float(score)
        if min_score is not None and s < min_score:
            continue
        results.append((s, docs[idx]))

    return results


def find_doc_by_exact_name(
    name: str,
    docs_path: Path = DOCS_PATH,
) -> Optional[Tuple[float, Dict[str, Any]]]:
    docs = load_docs(docs_path)
    n = (name or "").strip()
    if not n:
        return None

    low = n.lower()
    for d in docs:
        m = d.get("meta", {})
        if (m.get("korean_name") or "").strip() == n:
            return 1.0, d
        if (m.get("english_name") or "").strip().lower() == low:
            return 1.0, d
    return None


def find_pokemon_doc(
    name: str,
    k: int = 5,
    score_threshold: float = 0.35,
    index_path: Path = INDEX_PATH,
    docs_path: Path = DOCS_PATH,
) -> Optional[Tuple[float, Dict[str, Any]]]:
    exact = find_doc_by_exact_name(name, docs_path)
    if exact:
        return exact

    hits = query_rag_store(name, k=k, index_path=index_path, docs_path=docs_path)
    if not hits:
        return None

    score, doc = hits[0]
    if score < score_threshold:
        return None
    return score, doc


def batch_find_pokemon_docs(
    names: List[str],
    k: int = 5,
    score_threshold: float = 0.35,
    index_path: Path = INDEX_PATH,
    docs_path: Path = DOCS_PATH,
) -> List[Tuple[float, Dict[str, Any]]]:
    hits: List[Tuple[float, Dict[str, Any]]] = []
    for n in names:
        hit = find_pokemon_doc(
            n,
            k=k,
            score_threshold=score_threshold,
            index_path=index_path,
            docs_path=docs_path,
        )
        if hit:
            hits.append(hit)
    return hits
