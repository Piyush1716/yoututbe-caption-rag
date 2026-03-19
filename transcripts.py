"""
transcripts.py — Transcript fetching for the YouTube RAG pipeline
=================================================================
Supports two transcript sources:
  1. YouTube captions  (fast, free — via youtube-transcript-api)
  2. Sarvam AI STT     (any language, any video — via audio extraction + Sarvam batch API)
"""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


# ── YouTube Captions Path ─────────────────────────────────────────────────────

def fetch_transcript(video_id: str) -> str | None:
    """
    Fetch the transcript for a given YouTube video ID using YouTube's caption API.
    Returns plain text transcript or None if unavailable.
    """
    print(f"  [Transcript] Fetching YT captions for video: {video_id}")
    try:
        ytt_api         = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        transcript      = " ".join(chunk.text for chunk in transcript_list)
        word_count      = len(transcript.split())
        print(f"  [Transcript] ✅ Fetched successfully — {word_count} words")
        return transcript

    except TranscriptsDisabled:
        print(f"  [Transcript] ❌ No captions available for video: {video_id}")
        return None

    except Exception as e:
        print(f"  [Transcript] ❌ Error fetching transcript: {e}")
        return None


# ── Sarvam STT Path ───────────────────────────────────────────────────────────

def fetch_transcript_from_video(video_path: str) -> str:
    """
    Extract audio from a local video file and transcribe it via Sarvam AI.

    Pipeline:
        video file → audio (mp3) → Sarvam STT → plain text transcript

    This path supports ANY language and does NOT require YouTube captions.

    Args:
        video_path : Local path to the uploaded video file.

    Returns:
        Plain text transcript string.

    Raises:
        RuntimeError : If audio extraction or transcription fails.
    """
    # Import here so the YT caption path doesn't load heavy deps unnecessarily
    from audio_extractor import extract_audio, cleanup_audio
    from sarvam_stt import transcribe_audio

    print(f"  [Transcript] 🎬 Starting video → transcript pipeline for: {video_path}")

    # Step 1 — Extract audio
    audio_path = extract_audio(video_path)

    # Step 2 — Transcribe via Sarvam
    try:
        transcript = transcribe_audio(audio_path)
    finally:
        # Always clean up the temp audio file
        cleanup_audio(audio_path)

    return transcript


# ── Shared Chunking ───────────────────────────────────────────────────────────

def split_transcript(transcript: str, video_id: str) -> list:
    """
    Split a transcript into overlapping chunks and tag each with video_id metadata.
    Returns a list of LangChain Document objects.

    Works for transcripts from both the YouTube captions path and the Sarvam STT path.
    """
    print(f"  [Splitter] Splitting transcript into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.create_documents([transcript])

    # Tag every chunk with the video_id so we can filter by namespace later
    for chunk in chunks:
        chunk.metadata["video_id"] = video_id

    print(f"  [Splitter] ✅ Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks
