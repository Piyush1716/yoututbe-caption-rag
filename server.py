"""
server.py — FastAPI backend for Adaptive YouTube RAG
=====================================================

Run with:
    uvicorn server:app --reload --port 8000

Endpoints
---------
GET  /                          Health check
GET  /status/{video_id}         Check indexed levels for a video
POST /index                     Index a YouTube video (captions → 3 levels)
POST /index-video               Index any uploaded video file (Sarvam STT → 3 levels)
POST /chat                      Ask a question — auto-selects retrieval level
"""

import os
import uuid
import shutil
import traceback

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from transcripts import fetch_transcript, fetch_transcript_from_video, build_all_levels
from vectorstore import (
    get_pinecone_index, get_embedding_model,
    is_video_indexed, store_all_levels, get_indexed_levels,
)
from chain import adaptive_ask
from config import TEMP_AUDIO_DIR

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Adaptive YouTube RAG API", version="3.0.0")

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
    print("\n🚀 Starting Adaptive YouTube RAG server...")
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
    video_id:       str
    indexed:        bool
    indexed_levels: dict        # {"broad": bool, "medium": bool, "narrow": bool}
    message:        str


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/api_test")
def api_test():
    return FileResponse("static/api_test.html")


@app.get("/status/{video_id}", response_model=StatusResponse)
def check_status(video_id: str):
    """
    Check if a video is indexed and which levels are present.
    Returns a breakdown per level (broad / medium / narrow).
    """
    print(f"\n[Status] Checking video: {video_id}")
    indexed        = is_video_indexed(index, video_id)
    indexed_levels = get_indexed_levels(index, video_id)
    msg = "Fully indexed (all 3 levels)" if indexed else (
        f"Partially indexed: {indexed_levels}" if any(indexed_levels.values())
        else "Not indexed yet"
    )
    print(f"[Status] {msg}")
    return StatusResponse(
        video_id=video_id,
        indexed=indexed,
        indexed_levels=indexed_levels,
        message=msg,
    )


@app.post("/index")
def index_video(req: IndexRequest):
    """
    Index a YouTube video using its captions.

    Builds and stores THREE levels in Pinecone:
      • broad   — full-video LLM summary
      • medium  — per-section LLM summaries
      • narrow  — verbatim small chunks

    The video must have captions enabled. For caption-less or
    non-English videos, use POST /index-video instead.
    """
    video_id = req.video_id
    print(f"\n[Index] YouTube captions — video: {video_id}")

    try:
        if is_video_indexed(index, video_id):
            levels = get_indexed_levels(index, video_id)
            return {
                "video_id": video_id,
                "status":   "skipped",
                "message":  "Video already fully indexed",
                "levels":   levels,
            }

        # Fetch transcript
        transcript = fetch_transcript(video_id)
        if not transcript:
            raise HTTPException(
                status_code=404,
                detail="No captions available for this video. Try POST /index-video instead."
            )

        # Build all 3 levels
        level_docs = build_all_levels(transcript, video_id)

        # Store all 3 levels into Pinecone
        store_all_levels(level_docs, embedding, video_id)

        counts = {lvl: len(docs) for lvl, docs in level_docs.items()}
        print(f"[Index] ✅ Done — chunks per level: {counts}")
        return {
            "video_id":          video_id,
            "status":            "indexed",
            "transcript_source": "youtube_captions",
            "chunks_per_level":  counts,
            "message":           "Video indexed across all 3 retrieval levels",
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-video")
async def index_video_file(
    file:     UploadFile = File(...),
    video_id: str        = Form(None),
):
    """
    Index ANY video file via Sarvam AI speech-to-text.

    Supports any language — no captions required.
    Builds and stores the same three Pinecone levels as /index.

    Form fields:
        file      : Video file (mp4, mkv, avi, mov, webm, …)
        video_id  : Optional ID. Auto-generated from filename if omitted.
    """
    if not video_id:
        base     = os.path.splitext(file.filename)[0] if file.filename else "video"
        video_id = f"{base}_{uuid.uuid4().hex[:8]}"

    print(f"\n[IndexVideo] Upload — video_id: {video_id}, file: {file.filename}")

    ext        = os.path.splitext(file.filename)[-1] if file.filename else ".mp4"
    temp_video = os.path.join(TEMP_AUDIO_DIR, f"{video_id}{ext}")

    try:
        if is_video_indexed(index, video_id):
            return {
                "video_id": video_id,
                "status":   "skipped",
                "message":  "Video already fully indexed",
            }

        # Save the upload to disk temporarily
        print(f"  [IndexVideo] Saving upload → {temp_video}")
        with open(temp_video, "wb") as f:
            shutil.copyfileobj(file.file, f)
        size_mb = os.path.getsize(temp_video) / (1024 * 1024)
        print(f"  [IndexVideo] ✅ Saved ({size_mb:.1f} MB)")

        # Transcribe via Sarvam (audio extraction happens inside)
        transcript = fetch_transcript_from_video(temp_video)
        if not transcript or not transcript.strip():
            raise HTTPException(status_code=422, detail="Sarvam returned an empty transcript")

        # Build all 3 levels + store
        level_docs = build_all_levels(transcript, video_id)
        store_all_levels(level_docs, embedding, video_id)

        counts = {lvl: len(docs) for lvl, docs in level_docs.items()}
        print(f"[IndexVideo] ✅ Done — chunks per level: {counts}")
        return {
            "video_id":          video_id,
            "status":            "indexed",
            "transcript_source": "sarvam_stt",
            "chunks_per_level":  counts,
            "message":           "Video indexed via Sarvam STT across all 3 retrieval levels",
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)
            print(f"  [IndexVideo] 🗑️  Cleaned up: {temp_video}")


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Answer a question about an indexed video using Adaptive RAG.

    Automatically:
      1. Classifies the question (broad / medium / narrow)
      2. Retrieves from the matching Pinecone namespace
      3. Applies the appropriate prompt
      4. Returns the answer + which retrieval level was used
    """
    video_id = req.video_id
    question = req.question
    print(f"\n[Chat] video: {video_id} | question: '{question}'")

    try:
        if not is_video_indexed(index, video_id):
            raise HTTPException(
                status_code=400,
                detail="Video not indexed yet. Call POST /index or POST /index-video first.",
            )

        result = adaptive_ask(embedding, video_id=video_id, question=question)

        print(f"[Chat] ✅ Answered ({result['query_type']} level, {len(result['answer'])} chars)")
        return {
            "video_id":   video_id,
            "question":   question,
            "answer":     result["answer"],
            "query_type": result["query_type"],   # tells the client which level was used
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
