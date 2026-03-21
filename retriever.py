"""
retriever.py — Adaptive retriever that selects the right level based on query type
===================================================================================

Query type → Pinecone namespace → top-k

  broad   → "{video_id}__broad"   → top_k = 1   (single full-video summary)
  medium  → "{video_id}__medium"  → top_k = 4   (section summaries)
  narrow  → "{video_id}__narrow"  → top_k = 3   (verbatim raw chunks)
"""

from vectorstore import get_vectorstore
from config import NARROW_TOP_K, MEDIUM_TOP_K, BROAD_TOP_K

# ── Level → config mapping ────────────────────────────────────────────────────
_LEVEL_CONFIG = {
    "broad":  {"top_k": BROAD_TOP_K},
    "medium": {"top_k": MEDIUM_TOP_K},
    "narrow": {"top_k": NARROW_TOP_K},
}


def get_adaptive_retriever(embedding, video_id: str, query_type: str):
    """
    Build and return a retriever tuned for the given query type.

    Args:
        embedding  : The embedding model instance.
        video_id   : YouTube video ID to scope the search.
        query_type : One of "broad", "medium", "narrow" (from query_classifier).

    Returns:
        A LangChain retriever pointed at the correct Pinecone namespace.
    """
    if query_type not in _LEVEL_CONFIG:
        print(f"  [Retriever] ⚠️  Unknown query_type '{query_type}' — falling back to 'medium'")
        query_type = "medium"

    top_k = _LEVEL_CONFIG[query_type]["top_k"]
    print(f"  [Retriever] Building {query_type.upper()} retriever — video: {video_id}, top_k: {top_k}")

    vectorstore = get_vectorstore(embedding, video_id=video_id, level=query_type)
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    print(f"  [Retriever] ✅ Retriever ready")
    return retriever


def get_retriever(embedding, video_id: str = None):
    """
    Legacy helper — returns a NARROW retriever for backward compatibility.
    New code should use get_adaptive_retriever() instead.
    """
    print(f"  [Retriever] ⚠️  get_retriever() is deprecated. Use get_adaptive_retriever().")
    if video_id is None:
        raise ValueError("video_id is required for adaptive retrieval")
    return get_adaptive_retriever(embedding, video_id, query_type="narrow")


def debug_retrieval(embedding, video_id: str, question: str, query_type: str):
    """
    Dev utility — retrieve and print docs for a question at a given level.
    """
    retriever = get_adaptive_retriever(embedding, video_id, query_type)
    docs      = retriever.invoke(question)
    print(f"\n  [Retriever Debug] Level: {query_type} | Found {len(docs)} docs")
    for i, doc in enumerate(docs):
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"    [{i+1}] \"{preview}...\"")
    return docs
