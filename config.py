from dotenv import load_dotenv
import os

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SARVAM_API_KEY   = os.getenv("SARVAM_API_KEY")

# ── Pinecone Settings ─────────────────────────────────────────────────────────
PINECONE_INDEX_NAME = "youtube-rag"
PINECONE_CLOUD      = "aws"
PINECONE_REGION     = "us-east-1"
EMBEDDING_DIMENSION = 3072          # gemini-embedding-001

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"

# ── LLM Settings ─────────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"

# ── Adaptive Chunking — THREE levels stored at index time ─────────────────────
#
#   NARROW  → small raw chunks        (exact moment / timestamp lookups)
#   MEDIUM  → section summaries       (topic / speaker / concept lookups)
#   BROAD   → one full-video summary  (overview / "what is this about" questions)
#
NARROW_CHUNK_SIZE    = 500          # ≈ 30–60 s of speech per chunk
NARROW_CHUNK_OVERLAP = 100

MEDIUM_CHUNK_SIZE    = 6000         # ≈ 3–5 minute section per chunk (capped for Groq TPM)
MEDIUM_CHUNK_OVERLAP = 200

# BROAD: the entire transcript is summarised into ONE document — no chunking.

# ── Pinecone Namespace suffixes ───────────────────────────────────────────────
# Each video gets three namespaces:
#   "{video_id}__narrow"  /  "{video_id}__medium"  /  "{video_id}__broad"
NARROW_NS_SUFFIX = "__narrow"
MEDIUM_NS_SUFFIX = "__medium"
BROAD_NS_SUFFIX  = "__broad"

# ── Retriever top-k per level ─────────────────────────────────────────────────
NARROW_TOP_K = 3    # a few precise verbatim chunks
MEDIUM_TOP_K = 4    # several section summaries
BROAD_TOP_K  = 1    # the single whole-video summary document

# ── Sarvam STT ────────────────────────────────────────────────────────────────
SARVAM_STT_MODEL  = "saaras:v3"
SARVAM_OUTPUT_DIR = "./sarvam_output"
TEMP_AUDIO_DIR    = "./temp_audio"

# ── Chunking Settings ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200
