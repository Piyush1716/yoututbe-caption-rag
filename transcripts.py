from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def fetch_transcript(video_id: str) -> str | None:
    """
    Fetch the transcript for a given YouTube video ID.
    Returns plain text transcript or None if unavailable.
    """
    print(f"  [Transcript] Fetching transcript for video: {video_id}")
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id)
        transcript = " ".join(chunk.text for chunk in transcript_list)
        word_count = len(transcript.split())
        print(f"  [Transcript] ✅ Fetched successfully — {word_count} words")
        return transcript

    except TranscriptsDisabled:
        print(f"  [Transcript] ❌ No captions available for video: {video_id}")
        return None

    except Exception as e:
        print(f"  [Transcript] ❌ Error fetching transcript: {e}")
        return None


def split_transcript(transcript: str, video_id: str) -> list:
    """
    Split transcript into chunks and tag each with video_id metadata.
    Returns list of LangChain Document objects.
    """
    print(f"  [Splitter] Splitting transcript into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.create_documents([transcript])

    # Tag every chunk with the video_id so we can filter later
    for chunk in chunks:
        chunk.metadata["video_id"] = video_id

    print(f"  [Splitter] ✅ Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks
