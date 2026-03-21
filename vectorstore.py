"""
vectorstore.py — Pinecone vector store with three-level namespace management
============================================================================

Each video occupies THREE Pinecone namespaces:

    "{video_id}__narrow"  — verbatim small chunks
    "{video_id}__medium"  — LLM-summarised sections
    "{video_id}__broad"   — single full-video summary

The retriever selects the appropriate namespace based on query type,
ensuring the LLM always receives exactly the right amount of context.
"""

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD,
    PINECONE_REGION, EMBEDDING_DIMENSION, EMBEDDING_MODEL, GEMINI_API_KEY,
    NARROW_NS_SUFFIX, MEDIUM_NS_SUFFIX, BROAD_NS_SUFFIX,
)

os.environ["GEMINI_API_KEY"]   = GEMINI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _namespace(video_id: str, level: str) -> str:
    """Return the Pinecone namespace for a given video + level combination."""
    suffixes = {
        "narrow": NARROW_NS_SUFFIX,
        "medium": MEDIUM_NS_SUFFIX,
        "broad":  BROAD_NS_SUFFIX,
    }
    if level not in suffixes:
        raise ValueError(f"Unknown level '{level}'. Must be one of: narrow, medium, broad")
    return f"{video_id}{suffixes[level]}"


# ── Initialisation ────────────────────────────────────────────────────────────

def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Initialize and return the Google Generative AI embedding model."""
    print("  [Embeddings] Loading Google Generative AI embedding model...")
    embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"  [Embeddings] ✅ Model loaded: {EMBEDDING_MODEL}")
    return embedding


def get_pinecone_index():
    """
    Connect to Pinecone, creating the index if it doesn't exist yet.
    Returns the live Pinecone index object.
    """
    print("  [Pinecone] Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = pc.list_indexes().names()
    print(f"  [Pinecone] Existing indexes: {list(existing)}")

    if PINECONE_INDEX_NAME not in existing:
        print(f"  [Pinecone] Creating index '{PINECONE_INDEX_NAME}'...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print(f"  [Pinecone] ✅ Index created")
    else:
        print(f"  [Pinecone] ✅ Index '{PINECONE_INDEX_NAME}' already exists")

    index  = pc.Index(PINECONE_INDEX_NAME)
    stats  = index.describe_index_stats()
    total  = stats.get("total_vector_count", 0)
    print(f"  [Pinecone] ✅ Connected — {total} total vectors")
    return index


# ── Indexing status ───────────────────────────────────────────────────────────

def is_video_indexed(index, video_id: str) -> bool:
    """
    Return True if a video has been fully indexed (all three levels present).
    We check for the narrow namespace as the canonical signal — it is always
    created last in store_all_levels(), so its presence means the full pipeline ran.
    """
    stats       = index.describe_index_stats()
    namespaces  = stats.get("namespaces", {})
    narrow_ns   = _namespace(video_id, "narrow")
    return narrow_ns in namespaces


def get_indexed_levels(index, video_id: str) -> dict[str, bool]:
    """Return which levels are present for a video (useful for debugging)."""
    stats      = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})
    return {
        level: _namespace(video_id, level) in namespaces
        for level in ("broad", "medium", "narrow")
    }


# ── Storage ───────────────────────────────────────────────────────────────────

def store_chunks(chunks: list, embedding, video_id: str, level: str = "narrow"):
    """
    Store a list of Document chunks in Pinecone under the appropriate namespace.

    Args:
        chunks    : LangChain Document objects to embed and store.
        embedding : The embedding model instance.
        video_id  : YouTube video ID (used to build the namespace).
        level     : "narrow", "medium", or "broad".
    """
    ns = _namespace(video_id, level)
    print(f"  [VectorStore] Storing {len(chunks)} {level} chunks → namespace '{ns}'...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=PINECONE_INDEX_NAME,
        namespace=ns,
    )
    print(f"  [VectorStore] ✅ {level.capitalize()} chunks stored")


def store_all_levels(level_docs: dict, embedding, video_id: str):
    """
    Store all three levels of documents into their respective Pinecone namespaces.

    Args:
        level_docs : dict with keys "narrow", "medium", "broad" → list[Document]
        embedding  : The embedding model instance.
        video_id   : YouTube video ID.
    """
    print(f"\n  [VectorStore] 📦 Storing all 3 levels for video: {video_id}")
    # Store broad first, narrow last (narrow = completion signal for is_video_indexed)
    for level in ("broad", "medium", "narrow"):
        docs = level_docs.get(level, [])
        if docs:
            store_chunks(docs, embedding, video_id, level=level)
        else:
            print(f"  [VectorStore] ⚠️  No {level} docs to store — skipping")
    print(f"  [VectorStore] ✅ All levels stored for video: {video_id}")


# ── Retrieval ─────────────────────────────────────────────────────────────────

def get_vectorstore(embedding, video_id: str, level: str = "narrow") -> PineconeVectorStore:
    """
    Return a PineconeVectorStore scoped to a specific video + retrieval level.

    Args:
        embedding : The embedding model instance.
        video_id  : YouTube video ID.
        level     : "narrow", "medium", or "broad".

    Returns:
        A PineconeVectorStore pointed at the correct namespace.
    """
    ns = _namespace(video_id, level)
    print(f"  [VectorStore] ✅ VectorStore ready — video: {video_id}, level: {level}, ns: '{ns}'")
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding,
        namespace=ns,
    )
