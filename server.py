"""
server.py — FastAPI backend for YouTube RAG Chrome Extension
============================================================
Run with:
    uvicorn server:app --reload --port 8000

Endpoints
---------
GET  /                          Health check
GET  /status/{video_id}         Check if a video is indexed
POST /index                     Index a YouTube video via its captions
POST /index-video               Index ANY video via file upload + Sarvam STT  ← NEW
POST /chat                      Ask a question about an indexed video
"""

import os
import uuid
import shutil
import traceback

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transcripts import fetch_transcript, fetch_transcript_from_video, split_transcript
from vectorstore import (
    get_pinecone_index, get_embedding_model,
    is_video_indexed, store_chunks, get_vectorstore
)
from retriever import get_retriever
from chain import build_chain, ask
from config import TEMP_AUDIO_DIR

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="YouTube RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global singletons ──────────────────────────────────────────────────────────
embedding = None
index     = None

@app.on_event("startup")
async def startup():
    global embedding, index
    print("\n🚀 Starting YouTube RAG server...")
    embedding = get_embedding_model()
    index     = get_pinecone_index()
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
    print("✅ Server ready!\n")


# ── Request / Response Models ──────────────────────────────────────────────────
class IndexRequest(BaseModel):
    video_id: str

class ChatRequest(BaseModel):
    video_id: str
    question: str

class StatusResponse(BaseModel):
    video_id: str
    indexed:  bool
    message:  str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "YouTube RAG API v2 is running"}


@app.get("/status/{video_id}", response_model=StatusResponse)
def check_status(video_id: str):
    """Check if a video is already indexed in Pinecone."""
    print(f"\n[Status] Checking video: {video_id}")
    indexed = is_video_indexed(index, video_id)
    msg     = "Already indexed" if indexed else "Not indexed yet"
    return StatusResponse(video_id=video_id, indexed=indexed, message=msg)


@app.post("/index")
def index_video(req: IndexRequest):
    """
    Index a YouTube video using its auto-generated captions.
    Fast but requires the video to have captions enabled.
    """
    video_id = req.video_id
    print(f"\n[Index] YT captions — video: {video_id}")

    try:
        if is_video_indexed(index, video_id):
            return {"video_id": video_id, "status": "skipped", "message": "Video already indexed"}

        transcript = fetch_transcript(video_id)
        if not transcript:
            raise HTTPException(status_code=404, detail="No captions available for this video")

        chunks = split_transcript(transcript, video_id)
        store_chunks(chunks, embedding, video_id)

        return {
            "video_id": video_id,
            "status":   "indexed",
            "message":  f"Video indexed via captions ({len(chunks)} chunks)"
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-video")
async def index_video_file(
    file:     UploadFile = File(...),
    video_id: str        = Form(None)   # optional; auto-generated from filename if omitted
):
    """
    Index ANY video by uploading the file directly.

    Pipeline:
        uploaded file → audio extraction (moviepy) → Sarvam STT → chunk → Pinecone

    Supports any language — no captions required.

    Form fields:
        file      : The video file (mp4, mkv, avi, mov, webm, …)
        video_id  : Optional custom ID. If omitted, a UUID is generated.

    Returns:
        { video_id, status, message, transcript_source, chunks }
    """
    # ── Derive a stable video_id ───────────────────────────────────────────────
    if not video_id:
        base     = os.path.splitext(file.filename)[0] if file.filename else "video"
        video_id = f"{base}_{uuid.uuid4().hex[:8]}"

    print(f"\n[IndexVideo] File upload — video_id: {video_id}, file: {file.filename}")

    # ── Save uploaded file to a temp location ──────────────────────────────────
    ext          = os.path.splitext(file.filename)[-1] if file.filename else ".mp4"
    temp_video   = os.path.join(TEMP_AUDIO_DIR, f"{video_id}{ext}")

    try:
        # Skip if already indexed
        if is_video_indexed(index, video_id):
            return {
                "video_id": video_id,
                "status":   "skipped",
                "message":  "Video already indexed"
            }

        # Save upload to disk
        print(f"  [IndexVideo] Saving uploaded file → {temp_video}")
        with open(temp_video, "wb") as f:
            shutil.copyfileobj(file.file, f)
        size_mb = os.path.getsize(temp_video) / (1024 * 1024)
        print(f"  [IndexVideo] ✅ Saved ({size_mb:.1f} MB)")

        # Transcribe via Sarvam (audio extraction happens inside)
        print(f"  [IndexVideo] Starting Sarvam transcription pipeline...")
        transcript = fetch_transcript_from_video(temp_video)

        if not transcript or not transcript.strip():
            raise HTTPException(status_code=422, detail="Sarvam returned an empty transcript")

        # Chunk → store
        chunks = split_transcript(transcript, video_id)
        store_chunks(chunks, embedding, video_id)

        print(f"[IndexVideo] ✅ Done — {len(chunks)} chunks stored for video_id: {video_id}")
        return {
            "video_id":          video_id,
            "status":            "indexed",
            "transcript_source": "sarvam_stt",
            "chunks":            len(chunks),
            "message":           f"Video indexed via Sarvam STT ({len(chunks)} chunks)"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[IndexVideo] ❌ Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always delete the temp video file
        if os.path.exists(temp_video):
            os.remove(temp_video)
            print(f"  [IndexVideo] 🗑️  Cleaned up temp video: {temp_video}")


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Answer a question about an indexed video using RAG.
    Works for videos indexed via either /index (captions) or /index-video (Sarvam STT).
    """
    video_id = req.video_id
    question = req.question
    print(f"\n[Chat] Video: {video_id} | Question: '{question}'")

    try:
        if not is_video_indexed(index, video_id):
            raise HTTPException(
                status_code=400,
                detail="Video not indexed yet. Call /index or /index-video first."
            )

        retriever = get_retriever(embedding, video_id=video_id)
        chain     = build_chain(retriever)
        answer    = ask(chain, question)

        return {"video_id": video_id, "question": question, "answer": answer}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
