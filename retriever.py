from vectorstore import get_vectorstore
from config import TOP_K


def get_retriever(embedding, video_id: str = None):
    """
    Build and return a retriever.
    - If video_id is given  → searches only that video's namespace
    - If video_id is None   → searches across all indexed videos
    """
    scope = f"video '{video_id}'" if video_id else "ALL videos"
    print(f"  [Retriever] Building retriever — scope: {scope}, top_k={TOP_K}")

    vectorstore = get_vectorstore(embedding, video_id)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    print(f"  [Retriever] ✅ Retriever ready")
    return retriever


def test_retriever(retriever, question: str):
    """
    Test the retriever with a sample question and print how many docs were found.
    """
    print(f"  [Retriever] Testing with question: '{question}'")
    docs = retriever.invoke(question)
    print(f"  [Retriever] ✅ Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs):
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"    Doc {i+1}: \"{preview}...\"")
    return docs
