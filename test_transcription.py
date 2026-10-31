import os
import sys
from pathlib import Path

# Enable UTF-8 console output for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import FFMPEG_PATH, TEMP_DIR, SUBTITLES_DIR
from backend.transcriber import whisper_service, get_gpu_info, extract_audio_ffmpeg

def run_automated_test():
    print("========================================")
    print("WHISPER GPU SUBTITLE SYSTEM - AUTOMATED TEST")
    print("========================================")

    # 1. GPU Check
    gpu_info = get_gpu_info()
    print(f"[1] GPU Check:")
    print(f"    - Device: {gpu_info['device_name']}")
    print(f"    - VRAM: {gpu_info['vram_gb']} GB")
    print(f"    - CUDA Available: {gpu_info['cuda_available']}")
    print(f"    - Recommended Compute: {gpu_info['recommended_compute']}")
    print(f"    - FFmpeg Path: {FFMPEG_PATH}")
    assert os.path.exists(FFMPEG_PATH), f"FFmpeg not found at {FFMPEG_PATH}"

    # 2. Locate or create test sample
    env_sample = os.environ.get("SAMPLE_VIDEO")
    candidate_video = Path(env_sample) if env_sample else (BASE_DIR / "sample.mp4")
    test_video = None

    if candidate_video.exists():
        print(f"[2] Using existing user video: {candidate_video}")
        test_video = candidate_video
    else:
        # Create a 5-second test video with audio using FFmpeg
        test_video = TEMP_DIR / "synthetic_test.mp4"
        print(f"[2] Generating synthetic 5-second test video at {test_video}...")
        import subprocess
        cmd = [
            FFMPEG_PATH, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264", "-c:a", "aac",
            str(test_video)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert res.returncode == 0, f"Synthetic video creation failed: {res.stderr}"

    # 3. Test Audio Extraction with Noise Cancellation & Clarity Boost
    extracted_wav = TEMP_DIR / "test_extract.wav"
    print(f"[3] Testing FFmpeg audio extraction with AI Noise Cancellation to {extracted_wav}...")
    extract_audio_ffmpeg(test_video, extracted_wav, noise_preset="studio")
    assert extracted_wav.exists() and extracted_wav.stat().st_size > 0, "Audio extraction failed or empty"
    print(f"    - Extracted & Denoised WAV size: {extracted_wav.stat().st_size} bytes (OK)")
    # Delete test extracted file
    extracted_wav.unlink()

    # 4. End-to-End Transcription with GPU Whisper + Noise Cancellation
    job_id = "test_autojob"
    print(f"[4] Testing Whisper GPU transcription with Studio Noise Cancellation (job_id={job_id}, model=tiny)...")
    
    logged_steps = []
    def on_progress(p):
        step = p.get("step")
        pct = p.get("progress")
        msg = p.get("message")
        logged_steps.append((step, pct))
        print(f"    [Progress {pct}%] {msg}")

    result = whisper_service.transcribe_video(
        video_path=test_video,
        job_id=job_id,
        model_size="tiny",
        language=None,
        task="transcribe",
        noise_reduction="studio",
        progress_callback=on_progress
    )

    print("\n[5] Validating Transcription Results:")
    print(f"    - Noise Reduction: {result.get('noise_reduction')}")
    print(f"    - Clean Audio URL: {result.get('clean_audio_url')}")
    print(f"    - Detected language: {result['detected_language']}")
    print(f"    - Execution Device: {result['active_device']}")
    print(f"    - Compute Type: {result['active_compute']}")
    print(f"    - Subtitle Segments: {len(result['segments'])}")
    print(f"    - SRT File: {result['srt_file']}")
    print(f"    - VTT File: {result['vtt_file']}")
    print(f"    - TXT File: {result['txt_file']}")

    assert result.get('noise_reduction') == 'studio', "Noise reduction setting was not preserved"
    srt_path = Path(result['srt_file'])
    assert srt_path.exists(), "SRT file was not created!"
    print(f"\n--- Output SRT Content Preview ---\n{srt_path.read_text(encoding='utf-8')[:500]}\n----------------------------------")

    # 6. Verify intermediate file cleanup
    intermediate_audio = TEMP_DIR / f"{job_id}_audio.wav"
    assert not intermediate_audio.exists(), f"ERROR: Intermediate audio {intermediate_audio} was NOT deleted!"
    print(f"[6] Intermediate Cleanup Verification: Intermediate audio '{intermediate_audio.name}' was properly deleted! (PASSED)")

    print("\n>>> ALL AUTOMATED VERIFICATION TESTS PASSED SUCCESSFULLY! <<<\n")

if __name__ == "__main__":
    run_automated_test()
