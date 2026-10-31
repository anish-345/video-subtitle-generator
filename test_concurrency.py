import os
import sys
import subprocess
import concurrent.futures
import time
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from backend.config import FFMPEG_PATH, TEMP_DIR

def get_or_create_sample_video() -> Path:
    # 1. Environment variable override
    env_path = os.environ.get("SAMPLE_VIDEO")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 2. Check local workspace sample
    local_sample = BASE_DIR / "sample.mp4"
    if local_sample.exists():
        return local_sample

    # 3. Check data/uploads for any existing mp4
    uploads = list((BASE_DIR / "data" / "uploads").glob("*.mp4"))
    if uploads:
        return uploads[0]

    # 4. Generate synthetic 5-second test video
    synth_path = TEMP_DIR / "synthetic_concurrency_test.mp4"
    if not synth_path.exists():
        print(f"[Setup] Generating synthetic test video at {synth_path}...")
        cmd = [
            FFMPEG_PATH, "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=640x360:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
            "-c:v", "libx264", "-c:a", "aac",
            str(synth_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to generate synthetic test video: {res.stderr}")
    return synth_path

SAMPLE_VIDEO = get_or_create_sample_video()
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000")

def submit_and_track_client(client_id: int, model_name: str):
    print(f"[Client {client_id}] Uploading video with model='{model_name}' to Rust Gateway...")
    start_time = time.time()
    
    with open(SAMPLE_VIDEO, "rb") as f:
        res = requests.post(
            f"{BASE_URL}/api/transcribe",
            files={"file": (f"client_{client_id}.mp4", f, "video/mp4")},
            data={"model_size": model_name, "language": "auto", "task": "transcribe"},
            timeout=30
        )
    
    if res.status_code != 200:
        raise RuntimeError(f"[Client {client_id}] Upload failed: {res.text}")
    
    job_id = res.json()["job_id"]
    print(f"[Client {client_id}] Job {job_id} queued! Polling for GPU completion...")

    while True:
        status_res = requests.get(f"{BASE_URL}/api/jobs/{job_id}/status", timeout=10)
        if status_res.status_code == 200:
            data = status_res.json()
            st = data.get("status")
            if st == "completed":
                elapsed = time.time() - start_time
                srt_res = requests.get(f"{BASE_URL}/api/jobs/{job_id}/download/srt", timeout=10)
                srt_content = srt_res.text.strip()
                print(f"[Client {client_id}] SUCCESS! Finished in {elapsed:.1f}s. SRT Length: {len(srt_content)} chars.")
                return {
                    "client_id": client_id,
                    "job_id": job_id,
                    "elapsed": elapsed,
                    "segments": len(data.get("result", {}).get("segments", [])),
                    "srt_snippet": srt_content[:120]
                }
            elif st == "failed":
                raise RuntimeError(f"[Client {client_id}] Job failed: {data.get('error')}")
        time.sleep(1)

def run_concurrency_test():
    print("==================================================")
    print("  RUST MULTI-CLIENT CONCURRENCY VERIFICATION TEST ")
    print("==================================================")

    # 1. Check Rust status
    res = requests.get(f"{BASE_URL}/api/status")
    assert res.status_code == 200
    status_data = res.json()
    print("[Status Check] Rust Core Engine:", status_data["rust_core"]["engine"])
    print("[Status Check] GPU Device:", status_data["gpu"]["device_name"])

    # 2. Launch 3 concurrent clients in parallel
    clients = [
        (1, "tiny"),
        (2, "base"),
        (3, "base")
    ]

    print(f"\nSimulating {len(clients)} concurrent clients submitting at the exact same instant...")
    start_total = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(submit_and_track_client, cid, model) for cid, model in clients]
        results = [f.result() for f in futures]

    total_time = time.time() - start_total
    print("\n================ CONCURRENCY RESULTS ================")
    for r in results:
        print(f"  * Client {r['client_id']}: Job {r['job_id']} -> {r['segments']} segments generated in {r['elapsed']:.1f}s")
    print(f"Total time for 3 concurrent clients through Rust Queue: {total_time:.1f}s")
    print(">>> ALL CONCURRENT CLIENTS PROCESSED WITH ZERO GPU ERRORS! <<<\n")

if __name__ == "__main__":
    run_concurrency_test()
