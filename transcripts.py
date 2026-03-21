"""
transcripts.py — Transcript ingestion for the Adaptive YouTube RAG pipeline
============================================================================

Indexing builds THREE levels of representation for every video:

  NARROW  (raw chunks, ~500 chars)
      → verbatim transcript split into small overlapping windows
      → used for timestamp / exact-detail questions

  MEDIUM  (section summaries, ~3 000 chars → LLM-summarised)
      → transcript split into large sections, each summarised by the LLM
      → used for topic / speaker / concept questions

  BROAD   (one full-video summary)
      → entire transcript summarised into a single document
      → used for overview / "what is this video about" questions

Transcript sources
------------------
  • YouTube captions  (fast, free)          → fetch_transcript()
  • Sarvam AI STT     (any language / file) → fetch_transcript_from_video()
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,          # kept for backward compat (not used internally now)
    NARROW_CHUNK_SIZE, NARROW_CHUNK_OVERLAP,
    MEDIUM_CHUNK_SIZE, MEDIUM_CHUNK_OVERLAP,
    GROQ_API_KEY, LLM_MODEL,
)

# ── Token-budget constants ────────────────────────────────────────────────────
# Groq free tier: 12 000 TPM for llama-3.3-70b-versatile.
# ~4 chars ≈ 1 token, so we keep each LLM call well under 8 000 tokens of INPUT
# (leaving ~4 000 tokens headroom for the prompt wrapper + output).
#
# MAX_CHARS_PER_CALL = 8 000 tokens × 4 chars/token = 32 000 chars
# We use 6 000 tokens × 4 = 24 000 to be conservative.
_MAX_CHARS_PER_LLM_CALL = 24_000   # max transcript/section chars sent in one API call
_MEDIUM_SAFE_CHARS      = 6_000    # safe section size for medium summarisation

os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRANSCRIPT FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_transcript(video_id: str) -> str | None:
    """Fetch plain-text transcript for a YouTube video via its caption API."""
    print(f"  [Transcript] Fetching YT captions for video: {video_id}")
    try:
        ytt_api         = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        transcript      = " ".join(chunk.text for chunk in transcript_list)
        print(f"  [Transcript] ✅ {len(transcript.split())} words fetched")
        return transcript
    except TranscriptsDisabled:
        print(f"  [Transcript] ❌ Captions disabled for video: {video_id}")
        return None
    except Exception as e:
        print(f"  [Transcript] ❌ Error: {e}")
        return None


def fetch_transcript_from_video(video_path: str) -> str:
    """Extract audio from a local video file and transcribe via Sarvam AI."""
    from audio_extractor import extract_audio, cleanup_audio
    from sarvam_stt import transcribe_audio

    print(f"  [Transcript] 🎬 Video → transcript pipeline: {video_path}")
    audio_path = extract_audio(video_path)
    try:
        return transcribe_audio(audio_path)
    finally:
        cleanup_audio(audio_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. THREE-LEVEL SPLITTING
# ─────────────────────────────────────────────────────────────────────────────

def make_narrow_chunks(transcript: str, video_id: str) -> list[Document]:
    """
    Split transcript into small verbatim chunks (NARROW level).
    These are used for timestamp / exact-detail retrieval.
    """
    print(f"  [Splitter] Building NARROW chunks (size={NARROW_CHUNK_SIZE})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=NARROW_CHUNK_SIZE,
        chunk_overlap=NARROW_CHUNK_OVERLAP,
    )
    docs = splitter.create_documents([transcript])
    for i, doc in enumerate(docs):
        doc.metadata.update({"video_id": video_id, "level": "narrow", "chunk_index": i})
    print(f"  [Splitter] ✅ {len(docs)} narrow chunks")
    return docs


def make_medium_chunks(transcript: str, video_id: str) -> list[Document]:
    """
    Split transcript into sections and summarise each with the LLM (MEDIUM level).
    Section size is capped at _MEDIUM_SAFE_CHARS to stay within Groq TPM limits.
    """
    # Use the smaller of config value and our safe cap
    section_size = min(MEDIUM_CHUNK_SIZE, _MEDIUM_SAFE_CHARS)
    print(f"  [Splitter] Building MEDIUM chunks (section size={section_size})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=section_size,
        chunk_overlap=MEDIUM_CHUNK_OVERLAP,
    )
    sections = splitter.create_documents([transcript])
    print(f"  [Splitter] Summarising {len(sections)} sections with LLM...")

    llm  = ChatGroq(model=LLM_MODEL, temperature=0)
    docs = []

    for i, section in enumerate(sections):
        summary = _summarise_section(llm, section.page_content, i + 1, len(sections))
        doc = Document(
            page_content=summary,
            metadata={"video_id": video_id, "level": "medium", "section_index": i},
        )
        docs.append(doc)

    print(f"  [Splitter] ✅ {len(docs)} medium (section-summary) chunks")
    return docs


def make_broad_chunk(transcript: str, video_id: str) -> list[Document]:
    """
    Summarise the entire transcript into ONE document (BROAD level).
    Used for "what is this video about?" style questions.
    """
    print(f"  [Splitter] Building BROAD summary (full-video)...")
    llm     = ChatGroq(model=LLM_MODEL, temperature=0)
    summary = _summarise_full_video(llm, transcript)
    doc = Document(
        page_content=summary,
        metadata={"video_id": video_id, "level": "broad"},
    )
    print(f"  [Splitter] ✅ Full-video summary ({len(summary.split())} words)")
    return [doc]


def build_all_levels(transcript: str, video_id: str) -> dict[str, list[Document]]:
    """
    Build all three retrieval levels from a raw transcript.

    Returns a dict:
        {
            "narrow": [Document, ...],   # verbatim small chunks
            "medium": [Document, ...],   # LLM-summarised sections
            "broad":  [Document],        # single full-video summary
        }
    """
    print(f"\n  [Splitter] 🏗️  Building all 3 index levels for video: {video_id}")
    return {
        "narrow": make_narrow_chunks(transcript, video_id),
        "medium": make_medium_chunks(transcript, video_id),
        "broad":  make_broad_chunk(transcript, video_id),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. BACKWARD-COMPAT HELPER (still used by old /index path before upgrade)
# ─────────────────────────────────────────────────────────────────────────────

def split_transcript(transcript: str, video_id: str) -> list[Document]:
    """Legacy helper — returns narrow chunks only. Prefer build_all_levels()."""
    return make_narrow_chunks(transcript, video_id)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRIVATE LLM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _summarise_section(llm, section_text: str, idx: int, total: int) -> str:
    """
    Summarise one transcript section.
    If the section exceeds the safe char limit it is split and each piece
    summarised separately before being merged — keeping every call under the
    Groq TPM ceiling.
    """
    # If the section fits safely in one call, summarise directly
    if len(section_text) <= _MAX_CHARS_PER_LLM_CALL:
        summary = _single_summary_call(
            llm, section_text,
            instruction=(
                f"Summarise section {idx} of {total} of a video transcript in 3–6 sentences. "
                "Preserve names, numbers, key arguments, and specific claims."
            ),
        )
        print(f"    Section {idx}/{total}: {len(summary.split())} words")
        return summary

    # Section is too large — split it into safe pieces and merge
    print(f"    Section {idx}/{total}: too large ({len(section_text)} chars) — splitting...")
    pieces   = _split_text_safe(section_text)
    partials = [
        _single_summary_call(
            llm, piece,
            instruction=f"Summarise this part of section {idx}/{total} in 2–4 sentences.",
        )
        for piece in pieces
    ]
    merged = _merge_summaries(llm, partials, context=f"section {idx} of {total}")
    print(f"    Section {idx}/{total}: merged {len(pieces)} parts → {len(merged.split())} words")
    return merged


def _summarise_full_video(llm, transcript: str) -> str:
    """
    Produce a single comprehensive full-video summary using map-reduce.

    Strategy:
      1. Split the transcript into safe-sized pieces.
      2. Summarise each piece individually (map).
      3. Merge all partial summaries into one final summary (reduce).

    This guarantees no single LLM call exceeds the Groq TPM limit regardless
    of transcript length.
    """
    pieces = _split_text_safe(transcript)
    print(f"  [Splitter] Broad summary — {len(pieces)} piece(s) to summarise...")

    if len(pieces) == 1:
        # Short transcript — one call is enough
        summary = _single_summary_call(
            llm, pieces[0],
            instruction=(
                "Write a comprehensive summary (8–12 sentences) of this video transcript covering: "
                "the main topic, key arguments, important people/entities, and the overall conclusion. "
                "Preserve specific facts, numbers, and claims."
            ),
        )
        return summary

    # Map: summarise each piece
    print(f"  [Splitter] Map phase: summarising {len(pieces)} pieces...")
    partials = [
        _single_summary_call(
            llm, piece,
            instruction=(
                f"Summarise part {i+1} of {len(pieces)} of a video transcript in 4–6 sentences. "
                "Preserve names, numbers, key arguments, and specific claims."
            ),
        )
        for i, piece in enumerate(pieces)
    ]

    # Reduce: merge all partial summaries
    print(f"  [Splitter] Reduce phase: merging {len(partials)} partial summaries...")
    return _merge_summaries(llm, partials, context="the entire video")


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _split_text_safe(text: str) -> list[str]:
    """
    Split text into chunks that each fit safely within one LLM call.
    Uses a simple character splitter with no overlap (summaries don't need it).
    """
    if len(text) <= _MAX_CHARS_PER_LLM_CALL:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_MAX_CHARS_PER_LLM_CALL,
        chunk_overlap=0,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [doc.page_content for doc in splitter.create_documents([text])]


def _single_summary_call(llm, text: str, instruction: str) -> str:
    """Execute one summarisation call with a given instruction and text body."""
    prompt   = f"{instruction}\n\nText:\n{text}\n\nSummary:"
    response = llm.invoke(prompt)
    return response.content.strip()


def _merge_summaries(llm, summaries: list[str], context: str) -> str:
    """
    Merge a list of partial summaries into one coherent final summary.
    The merged input is always small (summaries are short), so this is
    guaranteed to fit within the token limit.
    """
    combined = "\n\n".join(f"Part {i+1}:\n{s}" for i, s in enumerate(summaries))
    prompt   = (
        f"You have summaries of different parts of {context}. "
        "Merge them into one comprehensive, coherent summary (8–12 sentences). "
        "Remove redundancy, preserve all key facts, names, and claims. "
        "Write in flowing prose.\n\n"
        f"{combined}\n\nMerged summary:"
    )
    response = llm.invoke(prompt)
    return response.content.strip()