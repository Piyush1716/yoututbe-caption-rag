"""
YouTube RAG — Main Entry Point
================================
Run this file to index videos and ask questions.

Usage:
    python main.py
"""

from transcripts import fetch_transcript, split_transcript
from vectorstore import get_pinecone_index, get_embedding_model, is_video_indexed, store_chunks
from retriever import get_retriever, test_retriever
from chain import build_chain, ask


# ── Videos to index ───────────────────────────────────────────────────────────
VIDEO_IDS = [
    "iys_pmJSp9M",   # Add as many video IDs as you want
    # "another_id",
]


def index_video(video_id: str, index, embedding):
    """
    Full pipeline to index a single video.
    Skips if already indexed.
    """
    print(f"\n{'='*55}")
    print(f"  Processing video: {video_id}")
    print(f"{'='*55}")

    # Check if already indexed
    if is_video_indexed(index, video_id):
        print(f"  ⏭️  Already indexed — skipping embedding step")
        return

    # Step 1 — Fetch transcript
    print("\n  📥 Step 1: Fetching transcript")
    transcript = fetch_transcript(video_id)
    if not transcript:
        print(f"  ❌ Could not fetch transcript for {video_id}. Skipping.")
        return

    # Step 2 — Split into chunks
    print("\n  ✂️  Step 2: Splitting transcript")
    chunks = split_transcript(transcript, video_id)

    # Step 3 — Store in Pinecone
    print("\n  📤 Step 3: Storing in Pinecone")
    store_chunks(chunks, embedding, video_id,)

    print(f"\n  ✅ Video '{video_id}' indexed successfully!\n")


def main():
    print("\n" + "🎬 " * 10)
    print("  YouTube RAG with Pinecone")
    print("🎬 " * 10 + "\n")

    # ── Setup ─────────────────────────────────────────────────────────────────
    print("── SETUP ──────────────────────────────────────────────")
    embedding = get_embedding_model()
    index     = get_pinecone_index()

    # ── Indexing ──────────────────────────────────────────────────────────────
    print("\n── INDEXING ────────────────────────────────────────────")
    for video_id in VIDEO_IDS:
        index_video(video_id, index, embedding)

    # ── RAG Query ─────────────────────────────────────────────────────────────
    print("\n── RAG QUERY ───────────────────────────────────────────")

    # Pick the video you want to query (or set to None to search all)
    target_video = VIDEO_IDS[0]

    print(f"\n  Building retriever for video: {target_video}")
    retriever = get_retriever(embedding, video_id=target_video)

    # Optional: test retriever before building full chain
    print("\n  Running retriever test...")
    test_retriever(retriever, question="What is this video about?")

    # Build chain
    print("\n  Building RAG chain...")
    chain = build_chain(retriever)

    # ── Ask questions ─────────────────────────────────────────────────────────
    print("\n── ANSWERS ─────────────────────────────────────────────")

    questions = [
        "Can you summarize the video?",
        "What are the main topics discussed?",
        # Add your own questions here
    ]

    for question in questions:
        answer = ask(chain, question)
        print(f"\n  💬 Q: {question}")
        print(f"  🤖 A: {answer}")
        print()


if __name__ == "__main__":
    main()
