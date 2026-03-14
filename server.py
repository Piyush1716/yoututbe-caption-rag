"""
server.py — FastAPI backend for YouTube RAG Chrome Extension
============================================================
Run with:
    uvicorn server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from transcripts import fetch_transcript, split_transcript
from vectorstore import get_pinecone_index, get_embedding_model, is_video_indexed, store_chunks, get_vectorstore
from retriever import get_retriever
from chain import build_chain, ask

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="YouTube RAG API", version="1.0.0")

# Allow Chrome extension to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Chrome extensions don't have a fixed origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global singletons (loaded once on startup) ─────────────────────────────────
embedding = None
index     = None

@app.on_event("startup")
async def startup():
    global embedding, index
    print("\n🚀 Starting YouTube RAG server...")
    print("  Loading embedding model...")
    embedding = get_embedding_model()
    print("  Connecting to Pinecone...")
    index = get_pinecone_index()
    print("✅ Server ready!\n")


# ── Request / Response Models ──────────────────────────────────────────────────
class IndexRequest(BaseModel):
    video_id: str

class ChatRequest(BaseModel):
    video_id: str
    question: str

class StatusResponse(BaseModel):
    video_id: str
    indexed: bool
    message: str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube RAG API is running"}


@app.get("/status/{video_id}", response_model=StatusResponse)
def check_status(video_id: str):
    """Check if a video is already indexed in Pinecone."""
    print(f"\n[Status] Checking video: {video_id}")
    indexed = is_video_indexed(index, video_id)
    msg = "Already indexed" if indexed else "Not indexed yet"
    print(f"[Status] {msg}")
    return StatusResponse(video_id=video_id, indexed=indexed, message=msg)


@app.post("/index")
def index_video(req: IndexRequest):
    """
    Index a YouTube video:
    - Fetches transcript
    - Splits into chunks
    - Stores in Pinecone (skips if already indexed)
    """
    video_id = req.video_id
    print(f"\n[Index] Request for video: {video_id}")

    try:
        # Skip if already indexed
        if is_video_indexed(index, video_id):
            print(f"[Index] ⏭️  Already indexed — skipping")
            return {"video_id": video_id, "status": "skipped", "message": "Video already indexed"}

        # Fetch transcript
        print(f"[Index] Fetching transcript...")
        transcript = fetch_transcript(video_id)
        if not transcript:
            raise HTTPException(status_code=404, detail="No transcript available for this video")

        # Split
        print(f"[Index] Splitting transcript...")
        chunks = split_transcript(transcript, video_id)

        # Store
        print(f"[Index] Storing in Pinecone...")
        store_chunks(chunks, embedding, video_id)

        print(f"[Index] ✅ Done — {len(chunks)} chunks stored")
        return {
            "video_id": video_id,
            "status": "indexed",
            "message": f"Video indexed successfully ({len(chunks)} chunks)"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Index] ❌ Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Answer a question about a specific video using RAG.
    The video must be indexed first via /index.
    """
    video_id = req.video_id
    question = req.question
    print(f"\n[Chat] Video: {video_id}")
    print(f"[Chat] Question: '{question}'")

    try:
        # Make sure video is indexed
        if not is_video_indexed(index, video_id):
            raise HTTPException(
                status_code=400,
                detail="Video not indexed yet. Call /index first."
            )

        # Build retriever + chain
        retriever = get_retriever(embedding, video_id=video_id)
        chain     = build_chain(retriever)

        # Get answer
        answer = ask(chain, question)
        print(f"[Chat] ✅ Answer generated ({len(answer)} chars)")
        return {"video_id": video_id, "question": question, "answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Chat] ❌ Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
