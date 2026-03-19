"""
sarvam_stt.py — Speech-to-text transcription using the Sarvam AI batch API
===========================================================================
Uses the saaras:v3 model with automatic language detection.
Returns the full transcript as a plain string.
"""

import os
import json
import glob
from sarvamai import SarvamAI
from config import SARVAM_API_KEY

# Directory where Sarvam writes its output JSON files
SARVAM_OUTPUT_DIR = "./sarvam_output"


def transcribe_audio(audio_path: str) -> str:
    """
    Send an audio file to Sarvam AI for transcription.

    Pipeline:
        1. Create a batch job (auto language detection, with speaker diarization)
        2. Upload the audio file
        3. Start the job and wait for completion
        4. Parse the output JSON and return the full transcript text

    Args:
        audio_path : Path to the audio file (.mp3 / .wav / etc.)

    Returns:
        Full transcript as a plain string.

    Raises:
        FileNotFoundError : If the audio file doesn't exist.
        RuntimeError      : If transcription fails or output cannot be parsed.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[SarvamSTT] ❌ Audio file not found: {audio_path}")

    print(f"  [SarvamSTT] Initialising Sarvam AI client...")
    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

    # ── Step 1: Create batch job ──────────────────────────────────────────────
    print(f"  [SarvamSTT] Creating batch transcription job...")
    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        mode="transcribe",
        language_code="unknown",      # auto-detect language
        with_diarization=True,
        num_speakers=2                # reasonable default; handles 1-speaker videos fine
    )
    print(f"  [SarvamSTT] ✅ Job created")

    # ── Step 2: Upload audio file ─────────────────────────────────────────────
    print(f"  [SarvamSTT] Uploading audio: {audio_path}")
    job.upload_files(file_paths=[audio_path])
    print(f"  [SarvamSTT] ✅ Upload complete")

    # ── Step 3: Start & wait ──────────────────────────────────────────────────
    print(f"  [SarvamSTT] Starting job and waiting for completion (this may take a while)...")
    job.start()
    job.wait_until_complete()
    print(f"  [SarvamSTT] ✅ Job completed")

    # ── Step 4: Check results ─────────────────────────────────────────────────
    file_results = job.get_file_results()
    successful   = file_results.get("successful", [])
    failed       = file_results.get("failed", [])

    if failed:
        for f in failed:
            print(f"  [SarvamSTT] ❌ Failed file: {f['file_name']} — {f.get('error_message', 'unknown error')}")

    if not successful:
        raise RuntimeError("[SarvamSTT] ❌ Transcription failed — no successful files returned by Sarvam")

    # ── Step 5: Download outputs ──────────────────────────────────────────────
    os.makedirs(SARVAM_OUTPUT_DIR, exist_ok=True)
    job.download_outputs(output_dir=SARVAM_OUTPUT_DIR)
    print(f"  [SarvamSTT] ✅ Downloaded {len(successful)} output file(s) to: {SARVAM_OUTPUT_DIR}")

    # ── Step 6: Parse JSON output → plain text ────────────────────────────────
    transcript = _parse_sarvam_output(SARVAM_OUTPUT_DIR, audio_path)
    word_count = len(transcript.split())
    print(f"  [SarvamSTT] ✅ Transcript extracted — {word_count} words")
    return transcript


def _parse_sarvam_output(output_dir: str, audio_path: str) -> str:
    """
    Parse Sarvam's JSON output file(s) and extract plain transcript text.

    Sarvam writes one JSON file per audio input, named after the audio file.
    The JSON contains a 'transcript' field (or a list of diarized segments).

    Args:
        output_dir : Directory where Sarvam placed the output JSON files.
        audio_path : Original audio path (used to find the matching JSON).

    Returns:
        Plain transcript string.
    """
    base_name    = os.path.splitext(os.path.basename(audio_path))[0]

    # Try to find the exact matching JSON first, then fall back to any JSON
    matching     = glob.glob(os.path.join(output_dir, f"{base_name}*.json"))
    all_jsons    = glob.glob(os.path.join(output_dir, "*.json"))
    candidates   = matching if matching else all_jsons

    if not candidates:
        raise RuntimeError(
            f"[SarvamSTT] ❌ No JSON output found in '{output_dir}'. "
            "The job may have completed but produced no output file."
        )

    # Use the most recently modified JSON if multiple exist
    json_path = max(candidates, key=os.path.getmtime)
    print(f"  [SarvamSTT] Parsing output: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return _extract_text_from_sarvam_json(data)


def _extract_text_from_sarvam_json(data: dict | list) -> str:
    """
    Extract a plain text transcript from Sarvam's JSON response.

    Handles two possible formats:
      A) { "transcript": "full text here" }
      B) { "segments": [ { "text": "...", "speaker": "...", ... }, ... ] }
      C) A list of segment dicts directly
    """
    # Format A — flat transcript string
    if isinstance(data, dict) and "transcript" in data:
        return data["transcript"].strip()

    # Format B — dict with a segments list
    if isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
        return " ".join(seg.get("text", "") for seg in segments).strip()

    # Format C — top-level list of segments
    if isinstance(data, list):
        return " ".join(
            item.get("text", item.get("transcript", ""))
            for item in data
        ).strip()

    # Fallback: dump everything as a string so we never lose data
    print("  [SarvamSTT] ⚠️  Unrecognised JSON format — returning raw dump")
    return json.dumps(data)
