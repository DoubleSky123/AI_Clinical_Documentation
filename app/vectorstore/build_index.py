"""
Vectorstore builder — ingests CDC/WHO clinical guidelines into FAISS.
Resume highlight: chunking strategy, MMR retrieval, incremental updates.

Usage:
    python -m app.vectorstore.build_index
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

INDEX_PATH = Path("vectorstore/faiss_index")
KB_PATH    = Path("data/medical_kb")

# ── Chunking strategy ────────────────────────────────────────────────────────
# 400-char chunks  → precise retrieval for dense clinical text
# 80-char overlap  → preserves cross-sentence clinical reasoning

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " "],
    length_function=len,
)


def _load_documents(kb_path: Path) -> list[Document]:
    """Load .txt guideline files and attach source metadata."""
    loader = DirectoryLoader(
        str(kb_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = Path(doc.metadata.get("source", "")).name
    logger.info("Loaded %d raw documents from %s", len(docs), kb_path)
    return docs


def build_vectorstore(kb_path: Path = KB_PATH) -> FAISS:
    """Full rebuild — run when adding new guideline documents."""
    docs   = _load_documents(kb_path)
    chunks = SPLITTER.split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vs = FAISS.from_documents(chunks, embeddings)

    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_PATH))
    logger.info("FAISS index saved → %s  (%d vectors)", INDEX_PATH, vs.index.ntotal)
    return vs


def update_vectorstore(new_docs_path: Path) -> FAISS:
    """
    Incremental update — merge new chunks into existing index.
    Avoids full rebuild when guidelines are updated in production.
    """
    vs         = load_vectorstore()
    new_docs   = _load_documents(new_docs_path)
    new_chunks = SPLITTER.split_documents(new_docs)
    vs.add_documents(new_chunks)
    vs.save_local(str(INDEX_PATH))
    logger.info("Added %d chunks; total vectors: %d", len(new_chunks), vs.index.ntotal)
    return vs


def load_vectorstore() -> FAISS:
    """Load persisted FAISS index from disk."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {INDEX_PATH}.\n"
            "Run:  python -m app.vectorstore.build_index"
        )
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vs = FAISS.load_local(
        str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
    )
    logger.info("Loaded FAISS index: %d vectors", vs.index.ntotal)
    return vs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_vectorstore()
