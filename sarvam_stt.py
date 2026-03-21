"""
sarvam_stt.py — Speech-to-text transcription using the Sarvam AI batch API
"""

import os
import json
import glob
from sarvamai import SarvamAI
from config import SARVAM_API_KEY

SARVAM_OUTPUT_DIR = "./sarvam_output"


def transcribe_audio(audio_path: str) -> str:
    """
    Send an audio file to Sarvam AI for batch transcription.
    Returns the full transcript as a plain string.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"[SarvamSTT] ❌ Audio file not found: {audio_path}")

    print(f"  [SarvamSTT] Initialising Sarvam AI client...")
    client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

    print(f"  [SarvamSTT] Creating batch job...")
    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        mode="transcribe",
        language_code="unknown",
        with_diarization=True,
        num_speakers=2,
    )

    print(f"  [SarvamSTT] Uploading: {audio_path}")
    job.upload_files(file_paths=[audio_path])

    print(f"  [SarvamSTT] Waiting for completion...")
    job.start()
    job.wait_until_complete()

    file_results = job.get_file_results()
    successful   = file_results.get("successful", [])
    failed       = file_results.get("failed", [])

    for f in failed:
        print(f"  [SarvamSTT] ❌ {f['file_name']}: {f.get('error_message')}")

    if not successful:
        raise RuntimeError("[SarvamSTT] ❌ No successful files returned")

    os.makedirs(SARVAM_OUTPUT_DIR, exist_ok=True)
    job.download_outputs(output_dir=SARVAM_OUTPUT_DIR)
    print(f"  [SarvamSTT] ✅ Downloaded outputs to: {SARVAM_OUTPUT_DIR}")

    return _parse_sarvam_output(SARVAM_OUTPUT_DIR, audio_path)


def _parse_sarvam_output(output_dir: str, audio_path: str) -> str:
    base_name  = os.path.splitext(os.path.basename(audio_path))[0]
    matching   = glob.glob(os.path.join(output_dir, f"{base_name}*.json"))
    all_jsons  = glob.glob(os.path.join(output_dir, "*.json"))
    candidates = matching if matching else all_jsons

    if not candidates:
        raise RuntimeError(f"[SarvamSTT] ❌ No JSON output found in '{output_dir}'")

    json_path = max(candidates, key=os.path.getmtime)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    transcript = _extract_text(data)
    print(f"  [SarvamSTT] ✅ {len(transcript.split())} words extracted")
    return transcript


def _extract_text(data) -> str:
    if isinstance(data, dict) and "transcript" in data:
        return data["transcript"].strip()
    if isinstance(data, dict) and "segments" in data:
        return " ".join(s.get("text", "") for s in data["segments"]).strip()
    if isinstance(data, list):
        return " ".join(i.get("text", i.get("transcript", "")) for i in data).strip()
    return json.dumps(data)
