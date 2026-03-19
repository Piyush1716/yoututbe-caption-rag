"""
audio_extractor.py — Extract audio from a video file using moviepy
"""

import os
from moviepy import VideoFileClip


def extract_audio(video_path: str, output_dir: str = "./temp_audio") -> str:
    """
    Extract audio from a video file and save it as an MP3.

    Args:
        video_path : Path to the input video file (any format moviepy supports).
        output_dir : Directory to save the extracted audio file.

    Returns:
        Path to the extracted audio MP3 file.

    Raises:
        FileNotFoundError : If the video file does not exist.
        RuntimeError      : If audio extraction fails.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[AudioExtractor] ❌ Video file not found: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Build output path — same filename, .mp3 extension
    base_name   = os.path.splitext(os.path.basename(video_path))[0]
    audio_path  = os.path.join(output_dir, f"{base_name}.mp3")

    print(f"  [AudioExtractor] Loading video: {video_path}")
    try:
        clip = VideoFileClip(video_path)

        if clip.audio is None:
            clip.close()
            raise RuntimeError(f"[AudioExtractor] ❌ Video has no audio track: {video_path}")

        print(f"  [AudioExtractor] Extracting audio → {audio_path}")
        clip.audio.write_audiofile(audio_path, logger=None)  # logger=None suppresses moviepy progress bar
        clip.close()

        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"  [AudioExtractor] ✅ Audio extracted ({size_mb:.2f} MB): {audio_path}")
        return audio_path

    except Exception as e:
        raise RuntimeError(f"[AudioExtractor] ❌ Failed to extract audio: {e}") from e


def cleanup_audio(audio_path: str):
    """Remove a temporary audio file after it has been processed."""
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"  [AudioExtractor] 🗑️  Cleaned up temp audio: {audio_path}")
