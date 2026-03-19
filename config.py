from dotenv import load_dotenv
import os

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SARVAM_API_KEY   = os.getenv("SARVAM_API_KEY")          # ← NEW: Sarvam AI STT key

# ── Pinecone Settings ─────────────────────────────────────────────────────────
PINECONE_INDEX_NAME = "youtube-rag"
PINECONE_CLOUD      = "aws"
PINECONE_REGION     = "us-east-1"
EMBEDDING_DIMENSION = 3072        # gemini-embedding-001 dimension

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"

# ── LLM Settings ─────────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"

# ── Chunking Settings ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# ── Retriever Settings ────────────────────────────────────────────────────────
TOP_K = 3

# ── Sarvam STT Settings ───────────────────────────────────────────────────────
SARVAM_STT_MODEL    = "saaras:v3"
SARVAM_OUTPUT_DIR   = "./sarvam_output"
TEMP_AUDIO_DIR      = "./temp_audio"
