from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_CLOUD,
    PINECONE_REGION, EMBEDDING_DIMENSION, EMBEDDING_MODEL, GEMINI_API_KEY
)
import os

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Initialize and return the Google embedding model."""
    print("  [Embeddings] Loading Google Generative AI embedding model...")
    embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"  [Embeddings] ✅ Model loaded: {EMBEDDING_MODEL}")
    return embedding


def get_pinecone_index():
    """
    Connect to Pinecone and ensure the index exists.
    Creates the index if it doesn't exist yet.
    Returns the Pinecone index object.
    """
    print("  [Pinecone] Connecting to Pinecone...")
    pc = Pinecone(api_key='pcsk_BzQqx_Py52qxfSrz4B71Lp1Wf6vtfXDhXq7xBr8odGzZxLmAyRrqRPiYEur3XypChNZAe')

    existing_indexes = pc.list_indexes().names()
    print(f"  [Pinecone] Existing indexes: {list(existing_indexes)}")

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"  [Pinecone] Index '{PINECONE_INDEX_NAME}' not found. Creating...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )
        print(f"  [Pinecone] ✅ Index '{PINECONE_INDEX_NAME}' created successfully")
    else:
        print(f"  [Pinecone] ✅ Index '{PINECONE_INDEX_NAME}' already exists")

    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)
    print(f"  [Pinecone] ✅ Connected — total vectors in index: {total_vectors}")
    return index


def is_video_indexed(index, video_id: str) -> bool:
    """
    Check if a video is already indexed by looking for its namespace.
    Namespace = video_id, so if the namespace exists, video is indexed.
    """
    stats = index.describe_index_stats()
    existing_namespaces = stats.get("namespaces", {})
    return video_id in existing_namespaces


def store_chunks(chunks: list, embedding, video_id: str):
    """
    Store document chunks in Pinecone under the video's namespace.
    """
    print(f"  [VectorStore] Storing {len(chunks)} chunks in namespace '{video_id}'...")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding,
        index_name=PINECONE_INDEX_NAME,
        namespace=video_id
    )
    print(f"  [VectorStore] ✅ Chunks stored successfully in namespace '{video_id}'")


def get_vectorstore(embedding, video_id: str = None) -> PineconeVectorStore:
    """
    Return a PineconeVectorStore scoped to a specific video namespace.
    If video_id is None, searches across all namespaces (all videos).
    """
    kwargs = {
        "index_name": PINECONE_INDEX_NAME,
        "embedding": embedding,
    }
    if video_id:
        kwargs["namespace"] = video_id
        print(f"  [VectorStore] ✅ VectorStore ready — scoped to video: {video_id}")
    else:
        print(f"  [VectorStore] ✅ VectorStore ready — searching across ALL videos")

    return PineconeVectorStore(**kwargs)
