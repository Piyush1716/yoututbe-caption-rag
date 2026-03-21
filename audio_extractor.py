"""
audio_extractor.py — Extract audio from a video file using moviepy
"""

import os
from moviepy import VideoFileClip


def extract_audio(video_path: str, output_dir: str = "./temp_audio") -> str:
    """
    Extract audio from a video file and save it as an MP3.
    Returns the path to the extracted audio file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[AudioExtractor] ❌ Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name  = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}.mp3")

    print(f"  [AudioExtractor] Loading video: {video_path}")
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            raise RuntimeError(f"[AudioExtractor] ❌ Video has no audio track: {video_path}")

        print(f"  [AudioExtractor] Extracting audio → {audio_path}")
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()

        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"  [AudioExtractor] ✅ Audio extracted ({size_mb:.2f} MB)")
        return audio_path
    except Exception as e:
        raise RuntimeError(f"[AudioExtractor] ❌ Extraction failed: {e}") from e


def cleanup_audio(audio_path: str):
    """Remove a temporary audio file after processing."""
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"  [AudioExtractor] 🗑️  Cleaned up: {audio_path}")
