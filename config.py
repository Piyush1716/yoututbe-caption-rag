from dotenv import load_dotenv
import os

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
# Either hardcode here (not recommended for production) or set as env variables
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY",)
GROQ_API_KEY    = os.getenv("GROQ_API_KEY",)
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")

# ── Pinecone Settings ─────────────────────────────────────────────────────────
PINECONE_INDEX_NAME = "youtube-rag"
PINECONE_CLOUD      = "aws"
PINECONE_REGION     = "us-east-1"
EMBEDDING_DIMENSION = 3072        # gemini-embedding-001 outputs 768-dim vectors

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"

# ── LLM Settings ─────────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"

# ── Chunking Settings ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

# ── Retriever Settings ────────────────────────────────────────────────────────
TOP_K = 3
