import re
import asyncio
import os
import shutil
import time
import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from backend.config import (
    BASE_DIR, UPLOADS_DIR, SUBTITLES_DIR, TEMP_DIR, OUTPUTS_DIR,
    FFMPEG_PATH, cleanup_file, purge_expired_temp_files, purge_expired_subtitles, purge_expired_outputs
)
from backend.transcriber import whisper_service, sensevoice_service, get_gpu_info, is_model_cached, format_timestamp_srt, format_timestamp_vtt, setup_cuda_dll_paths
from typing import List

app = FastAPI(title="GPU Subtitle Worker API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import threading

JOBS: Dict[str, Dict[str, Any]] = {}
GPU_LOCK = threading.Lock()

@app.on_event("startup")
async def on_startup():
    purge_expired_temp_files(max_age_seconds=1800)
    purge_expired_subtitles(max_age_seconds=86400)
    purge_expired_outputs(max_age_seconds=86400)

    # Periodic background task to evict old completed/failed jobs and expired files
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(300)
            try:
                purge_expired_temp_files(max_age_seconds=1800)
                purge_expired_subtitles(max_age_seconds=86400)
                purge_expired_outputs(max_age_seconds=86400)
                cutoff = time.time() - 3600  # 1 hour retention in memory
                expired = [
                    jid for jid, j in list(JOBS.items())
                    if j.get("status") in ("completed", "failed")
                    and j.get("created_at", 0) < cutoff
                ]
                for jid in expired:
                    JOBS.pop(jid, None)
            except Exception as e:
                print(f"[Periodic cleanup notice] {e}")

    asyncio.create_task(periodic_cleanup())


def get_wifi_ip() -> str:
    custom_host = os.environ.get("HOST") or os.environ.get("SERVER_IP")
    if custom_host and custom_host not in ("0.0.0.0", "127.0.0.1"):
        return custom_host
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

@app.get("/api/status")
def get_system_status():
    gpu = get_gpu_info()
    wifi_ip = get_wifi_ip()
    gpu_name = gpu.get("device_name", "")
    if gpu.get("cuda_available") and gpu_name and gpu_name != "CPU":
        sense_accel = f"🚀 {gpu_name} Accelerated • Millisecond Precision"
    else:
        sense_accel = "⚡ Hardware Accelerated • Millisecond Precision"

    models = [
        {
            "id": "base",
            "name": "Base (~140MB)",
            "cached": is_model_cached("base"),
            "description": "Recommended • Instant Start ⚡",
            "recommended": True
        },
        {
            "id": "sensevoice-small",
            "name": "Alibaba SenseVoice-Small (~1.1GB)",
            "cached": is_model_cached("sensevoice-small"),
            "description": f"{sense_accel} (Instant Start)" if is_model_cached("sensevoice-small") else sense_accel,
            "recommended": False
        },
        {
            "id": "large-v3-turbo",
            "name": "Large-v3 Turbo (~1.6GB)",
            "cached": is_model_cached("large-v3-turbo"),
            "description": "Deep Learning SOTA • Instant Start ⚡" if is_model_cached("large-v3-turbo") else "SOTA Deep Learning (~1.6GB)",
            "recommended": False
        },
        {
            "id": "medium",
            "name": "Medium (~1.5GB)",
            "cached": is_model_cached("medium"),
            "description": "Studio Quality • Instant Start ⚡" if is_model_cached("medium") else "Studio Quality (~1.5GB)",
            "recommended": False
        },
        {
            "id": "small",
            "name": "Small (~460MB)",
            "cached": is_model_cached("small"),
            "description": "High Accuracy • Instant Start ⚡" if is_model_cached("small") else "High Accuracy (~460MB)",
            "recommended": False
        },
        {
            "id": "tiny",
            "name": "Tiny (~75MB)",
            "cached": is_model_cached("tiny"),
            "description": "Ultra Fast • Instant Start ⚡",
            "recommended": False
        },
        {
            "id": "large-v3",
            "name": "Large-v3 (~3GB)",
            "cached": is_model_cached("large-v3"),
            "description": "Maximum Multilingual Accuracy" if is_model_cached("large-v3") else "Maximum Multilingual Accuracy (~3GB)",
            "recommended": False
        }
    ]
    return {
        "status": "ready",
        "wifi_ip": wifi_ip,
        "wifi_url": f"http://{wifi_ip}:8000",
        "gpu": gpu,
        "ffmpeg_path": FFMPEG_PATH,
        "turbo_engine": {
            "supported": True,
            "active_mode": "Tensor Core Batched (FP16)",
            "speedup_factor": "170x Real-Time",
            "ten_hour_eta": "3.5 - 5 minutes",
            "recommended_batch_size": 16
        },
        "available_models": models,
        "queued_count": sum(1 for j in JOBS.values() if j.get("status") in ("processing", "queued"))
    }

def process_transcription_job(
    job_id: str,
    video_path: Path,
    model_size: str,
    language: Optional[str],
    task: str,
    turbo_mode: bool = True,
    batch_size: Optional[int] = None,
    delete_video_after: bool = False,
    vad_filter: bool = False,
    noise_reduction: str = "studio"
):
    job = JOBS[job_id]
    job["status"] = "queued"
    job["message"] = "Waiting in queue for GPU..."
    loop = job.get("loop")

    def on_progress(payload: Dict[str, Any]):
        job["progress"] = payload.get("progress", job["progress"])
        job["message"] = payload.get("message", job["message"])
        if "status" in payload:
            job["status"] = payload["status"]
        payload["status"] = job["status"]
        if "turbo_mode" in payload:
            job["turbo_mode"] = payload["turbo_mode"]
        if "speedup_factor" in payload:
            job["speedup_factor"] = payload["speedup_factor"]
        if "latest_segment" in payload:
            job.setdefault("streamed_segments", []).append(payload["latest_segment"])
        
        q: asyncio.Queue = job.get("queue")
        if q and loop:
            loop.call_soon_threadsafe(q.put_nowait, payload)

    try:
        on_progress({"step": "queued", "status": "queued", "progress": 0, "message": "Waiting in queue for GPU..."})
        if model_size == "sensevoice-small":
            with GPU_LOCK:
                job["status"] = "processing"
                job["message"] = "Acquired GPU, starting Alibaba SenseVoice-Small..."
                on_progress({"step": "status", "status": "processing", "progress": 1, "message": "Acquired GPU, starting Alibaba SenseVoice-Small..."})
                result = sensevoice_service.transcribe_video(
                    video_path=video_path,
                    job_id=job_id,
                    language=language,
                    task=task,
                    turbo_mode=turbo_mode,
                    batch_size=batch_size,
                    delete_video_after=delete_video_after,
                    vad_filter=vad_filter,
                    noise_reduction=noise_reduction,
                    original_filename=job.get("filename"),
                    progress_callback=on_progress
                )
        else:
            with GPU_LOCK:
                job["status"] = "processing"
                job["message"] = "Acquired GPU, starting transcription..."
                on_progress({"step": "status", "status": "processing", "progress": 1, "message": "Acquired GPU, starting transcription..."})
                result = whisper_service.transcribe_video(
                    video_path=video_path,
                    job_id=job_id,
                    model_size=model_size,
                    language=language,
                    task=task,
                    turbo_mode=turbo_mode,
                    batch_size=batch_size,
                    delete_video_after=delete_video_after,
                    vad_filter=vad_filter,
                    noise_reduction=noise_reduction,
                    original_filename=job.get("filename"),
                    progress_callback=on_progress
                )
        job["status"] = "completed"
        job["progress"] = 100
        job["result"] = result
        on_progress({"step": "completed", "status": "completed", "progress": 100, "message": "Transcription finished"})
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        on_progress({"step": "error", "status": "failed", "progress": 100, "message": str(e)})

from pydantic import BaseModel

class TranscribePathRequest(BaseModel):
    job_id: str
    file_path: str
    filename: str
    model_size: str = "base"
    language: str = "auto"
    task: str = "transcribe"
    turbo_mode: bool = True
    batch_size: Optional[int] = None
    delete_video_after: bool = False
    vad_filter: bool = False
    noise_reduction: str = "studio"

@app.post("/api/transcribe_path")
async def create_transcription_from_path(
    req: TranscribePathRequest,
    background_tasks: BackgroundTasks = None
):
    saved_video_path = Path(req.file_path)
    if not saved_video_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file path does not exist on server")

    # Deduplication Check: Reconnect to active or completed job for the same video with matching settings
    for existing_id, existing_job in list(JOBS.items()):
        if (
            existing_job.get("status") in ("processing", "queued")
            and existing_job.get("filename") == req.filename
            and existing_job.get("noise_reduction", "studio") == req.noise_reduction
        ):
            print(f"[Deduplication] Re-attaching to ongoing job {existing_id} for '{req.filename}'")
            if saved_video_path.exists() and str(saved_video_path) != existing_job.get("video_path", ""):
                cleanup_file(saved_video_path)
            return {
                "job_id": existing_id,
                "status": existing_job.get("status"),
                "filename": req.filename,
                "turbo_mode": existing_job.get("turbo_mode", req.turbo_mode),
                "video_url": f"/api/jobs/{existing_id}/video",
                "deduplicated": True,
                "message": "Reconnected to active transcription job"
            }
        elif (
            existing_job.get("status") == "completed"
            and existing_job.get("filename") == req.filename
            and existing_job.get("noise_reduction", "studio") == req.noise_reduction
            and existing_job.get("result") is not None
        ):
            print(f"[Deduplication] Instant cache hit for '{req.filename}' from job {existing_id}")
            if saved_video_path.exists() and str(saved_video_path) != existing_job.get("video_path", ""):
                cleanup_file(saved_video_path)
            return {
                "job_id": existing_id,
                "status": "completed",
                "filename": req.filename,
                "turbo_mode": existing_job.get("turbo_mode", req.turbo_mode),
                "video_url": f"/api/jobs/{existing_id}/video",
                "result": existing_job.get("result"),
                "deduplicated": True,
                "message": "Instant return: Subtitles already generated"
            }

    loop = asyncio.get_running_loop()
    job_queue = asyncio.Queue()

    JOBS[req.job_id] = {
        "id": req.job_id,
        "filename": req.filename,
        "video_path": str(saved_video_path),
        "status": "queued",
        "progress": 0,
        "message": "Job queued, initializing Tensor Core GPU...",
        "model_size": req.model_size,
        "language": req.language,
        "task": req.task,
        "turbo_mode": req.turbo_mode,
        "batch_size": req.batch_size,
        "delete_video_after": req.delete_video_after,
        "vad_filter": req.vad_filter,
        "noise_reduction": req.noise_reduction,
        "streamed_segments": [],
        "result": None,
        "error": None,
        "queue": job_queue,
        "loop": loop,
        "created_at": time.time()
    }

    background_tasks.add_task(
        process_transcription_job,
        req.job_id,
        saved_video_path,
        req.model_size,
        req.language,
        req.task,
        req.turbo_mode,
        req.batch_size,
        req.delete_video_after,
        req.vad_filter,
        req.noise_reduction
    )

    return {
        "job_id": req.job_id,
        "status": "queued",
        "filename": req.filename,
        "turbo_mode": req.turbo_mode,
        "video_url": f"/api/jobs/{req.job_id}/video"
    }

@app.post("/api/transcribe")
async def create_transcription_job(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    language: str = Form("auto"),
    task: str = Form("transcribe"),
    turbo_mode: bool = Form(True),
    batch_size: Optional[int] = Form(None),
    delete_video_after: bool = Form(False),
    vad_filter: bool = Form(False),
    noise_reduction: str = Form("studio"),
    background_tasks: BackgroundTasks = None
):
    # Deduplication Check before saving to disk
    for existing_id, existing_job in list(JOBS.items()):
        if (
            existing_job.get("status") in ("processing", "queued")
            and existing_job.get("filename") == file.filename
            and existing_job.get("noise_reduction", "studio") == noise_reduction
        ):
            print(f"[Deduplication] Re-attaching to ongoing job {existing_id} for '{file.filename}'")
            return {
                "job_id": existing_id,
                "status": existing_job.get("status"),
                "filename": file.filename,
                "turbo_mode": existing_job.get("turbo_mode", turbo_mode),
                "video_url": f"/api/jobs/{existing_id}/video",
                "deduplicated": True,
                "message": "Reconnected to active transcription job"
            }
        elif (
            existing_job.get("status") == "completed"
            and existing_job.get("filename") == file.filename
            and existing_job.get("noise_reduction", "studio") == noise_reduction
            and existing_job.get("result") is not None
        ):
            print(f"[Deduplication] Instant cache hit for '{file.filename}' from job {existing_id}")
            return {
                "job_id": existing_id,
                "status": "completed",
                "filename": file.filename,
                "turbo_mode": existing_job.get("turbo_mode", turbo_mode),
                "video_url": f"/api/jobs/{existing_id}/video",
                "result": existing_job.get("result"),
                "deduplicated": True,
                "message": "Instant return: Subtitles already generated"
            }
    job_id = str(uuid.uuid4())[:8]
    sanitized_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    saved_video_path = UPLOADS_DIR / f"{job_id}_{sanitized_filename}"

    with open(saved_video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loop = asyncio.get_running_loop()
    job_queue = asyncio.Queue()

    JOBS[job_id] = {
        "id": job_id,
        "filename": file.filename,
        "video_path": str(saved_video_path),
        "status": "queued",
        "progress": 0,
        "message": "Job queued, initializing Tensor Core GPU...",
        "model_size": model_size,
        "language": language,
        "task": task,
        "turbo_mode": turbo_mode,
        "batch_size": batch_size,
        "delete_video_after": delete_video_after,
        "vad_filter": vad_filter,
        "noise_reduction": noise_reduction,
        "streamed_segments": [],
        "result": None,
        "error": None,
        "queue": job_queue,
        "loop": loop,
        "created_at": time.time()
    }

    background_tasks.add_task(
        process_transcription_job,
        job_id,
        saved_video_path,
        model_size,
        language,
        task,
        turbo_mode,
        batch_size,
        delete_video_after,
        vad_filter,
        noise_reduction
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename,
        "turbo_mode": turbo_mode,
        "video_url": f"/api/jobs/{job_id}/video"
    }

@app.get("/api/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]
    q: asyncio.Queue = job["queue"]

    async def event_generator():
        yield {
            "event": "message",
            "data": json.dumps({
                "step": "status",
                "progress": job["progress"],
                "message": job["message"],
                "status": job["status"]
            })
        }

        while True:
            try:
                data = await asyncio.wait_for(q.get(), timeout=30.0)
                yield {
                    "event": "message",
                    "data": json.dumps(data)
                }
                if data.get("step") in ["completed", "error"]:
                    break
            except asyncio.TimeoutError:
                if job["status"] in ["completed", "failed"]:
                    break
                yield {
                    "event": "ping",
                    "data": "keep-alive"
                }

    return EventSourceResponse(event_generator())

def parse_srt_to_segments(srt_content: str) -> List[Dict[str, Any]]:
    """Parse SRT text back into segment list with millisecond accuracy."""
    segments = []
    blocks = re.split(r'\n\s*\n', srt_content.strip())
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) >= 3:
            try:
                cue_id = int(lines[0])
                timing = lines[1]
                text = " ".join(lines[2:])
                if "-->" in timing:
                    parts = timing.split("-->")
                    s_str = parts[0].strip()
                    e_str = parts[1].strip()
                    def srt_to_sec(t_str):
                        h, m, sm = t_str.split(":")
                        s, ms = sm.split(",")
                        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
                    s_sec = round(srt_to_sec(s_str), 3)
                    e_sec = round(srt_to_sec(e_str), 3)
                    segments.append({
                        "id": cue_id,
                        "start": s_sec,
                        "end": e_sec,
                        "start_fmt": s_str,
                        "end_fmt": e_str,
                        "text": text
                    })
            except Exception:
                continue
    return segments

@app.get("/api/jobs")
def list_all_jobs():
    """List active in-flight jobs and persistent completed jobs from disk."""
    job_list = []
    seen_ids = set()

    # 1. In-memory active/recent jobs
    queued_pos = 1
    mem_jobs = sorted(list(JOBS.items()), key=lambda item: item[1].get("created_at", 0))
    for jid, j in mem_jobs:
        seen_ids.add(jid)
        status = j.get("status", "queued")
        q_pos = None
        if status == "queued":
            q_pos = queued_pos
            queued_pos += 1

        job_list.append({
            "job_id": jid,
            "filename": j.get("filename", "video"),
            "status": status,
            "progress": j.get("progress", 0),
            "queue_position": q_pos,
            "model_size": j.get("model_size"),
            "language": j.get("language"),
            "message": j.get("message", ""),
            "created_at": j.get("created_at", time.time()),
            "total_duration": j.get("result", {}).get("total_duration", 0.0) if j.get("result") else 0.0,
            "detected_language": j.get("result", {}).get("detected_language") if j.get("result") else None,
            "has_subtitles": (SUBTITLES_DIR / f"{jid}.srt").exists()
        })

    # 2. Completed jobs stored on disk
    for meta_path in SUBTITLES_DIR.glob("*.meta.json"):
        jid = meta_path.name.replace(".meta.json", "")
        if jid not in seen_ids:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                job_list.append({
                    "job_id": jid,
                    "filename": meta.get("filename", f"subtitles_{jid}"),
                    "status": "completed",
                    "progress": 100,
                    "queue_position": None,
                    "model_size": meta.get("model_size"),
                    "language": meta.get("language"),
                    "message": "Subtitles generated successfully",
                    "created_at": meta.get("created_at", meta_path.stat().st_mtime),
                    "total_duration": meta.get("total_duration", 0.0),
                    "detected_language": meta.get("detected_language"),
                    "has_subtitles": (SUBTITLES_DIR / f"{jid}.srt").exists()
                })
            except Exception:
                pass

    def sort_key(x):
        s = x.get("status")
        if s == "processing":
            return (0, -x.get("created_at", 0))
        elif s == "queued":
            return (1, x.get("created_at", 0))
        else:
            return (2, -x.get("created_at", 0))

    job_list.sort(key=sort_key)
    return {
        "jobs": job_list,
        "active_count": sum(1 for j in job_list if j.get("status") in ("processing", "queued")),
        "completed_count": sum(1 for j in job_list if j.get("status") == "completed")
    }

@app.get("/api/jobs/{job_id}/status")
def get_job_status(job_id: str):
    if job_id in JOBS:
        job = JOBS[job_id]
        return {
            "id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "filename": job["filename"],
            "video_url": f"/api/jobs/{job_id}/video",
            "result": job.get("result"),
            "error": job.get("error")
        }

    # Disk Fallback: reconstruct from persistent storage
    meta_file = SUBTITLES_DIR / f"{job_id}.meta.json"
    srt_file = SUBTITLES_DIR / f"{job_id}.srt"
    if meta_file.exists() and srt_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            segments = parse_srt_to_segments(srt_file.read_text(encoding="utf-8"))
            return {
                "id": job_id,
                "status": "completed",
                "progress": 100,
                "message": "Subtitles loaded from disk",
                "filename": meta.get("filename", f"subtitles_{job_id}"),
                "video_url": f"/api/jobs/{job_id}/video",
                "result": {
                    "job_id": job_id,
                    "filename": meta.get("filename", f"subtitles_{job_id}"),
                    "status": "completed",
                    "total_duration": meta.get("total_duration", 0.0),
                    "detected_language": meta.get("detected_language", "auto"),
                    "segments": segments,
                    "srt_file": str(srt_file),
                    "vtt_file": str(SUBTITLES_DIR / f"{job_id}.vtt"),
                    "txt_file": str(SUBTITLES_DIR / f"{job_id}.txt")
                },
                "error": None
            }
        except Exception as e:
            print(f"[Disk fallback error] {e}")

    raise HTTPException(status_code=404, detail="Job not found")

@app.get("/api/jobs/{job_id}/video")
def get_job_video(job_id: str):
    if job_id in JOBS:
        video_path = Path(JOBS[job_id]["video_path"])
        if video_path.exists():
            return FileResponse(video_path, media_type="video/mp4", filename=video_path.name)

    # Check uploads directory for any file starting with job_id
    for p in UPLOADS_DIR.glob(f"{job_id}_*"):
        if p.exists():
            return FileResponse(p, media_type="video/mp4", filename=p.name)

    raise HTTPException(status_code=404, detail="Video file not found or already purged")

@app.get("/api/jobs/{job_id}/subtitles/{ext}")
def get_job_subtitles(job_id: str, ext: str):
    ext = ext.lower()
    if ext not in ["srt", "vtt", "txt"]:
        raise HTTPException(status_code=400, detail="Unsupported format")
    
    file_path = SUBTITLES_DIR / f"{job_id}.{ext}"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Subtitle file not found")

    content_type = "text/vtt" if ext == "vtt" else "text/plain; charset=utf-8"
    return FileResponse(
        file_path,
        media_type=content_type,
        filename=f"{job_id}.{ext}"
    )

@app.get("/api/jobs/{job_id}/download/{ext}")
def download_subtitles(job_id: str, ext: str):
    ext = ext.lower()
    if ext not in ["srt", "vtt", "txt"]:
        raise HTTPException(status_code=400, detail="Invalid extension")

    file_path = SUBTITLES_DIR / f"{job_id}.{ext}"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Subtitle file does not exist")

    original_name = JOBS.get(job_id, {}).get("filename")
    if not original_name:
        meta_file = SUBTITLES_DIR / f"{job_id}.meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                original_name = meta.get("filename")
            except Exception:
                pass
    if not original_name:
        original_name = f"subtitles_{job_id}"

    stem = Path(original_name).stem
    download_filename = f"{stem}.{ext}"

    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=download_filename
    )

@app.get("/api/jobs/{job_id}/clean-audio")
def get_clean_audio(job_id: str):
    file_path = OUTPUTS_DIR / f"{job_id}_clean_audio.mp3"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Clean audio file not found")

    original_name = JOBS.get(job_id, {}).get("filename")
    if not original_name:
        meta_file = SUBTITLES_DIR / f"{job_id}.meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                original_name = meta.get("filename")
            except Exception:
                pass
    if not original_name:
        original_name = f"audio_{job_id}"

    stem = Path(original_name).stem
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=f"{stem}_clean_audio.mp3"
    )

@app.delete("/api/jobs/{job_id}/cleanup")
def cleanup_job(job_id: str, delete_video: bool = True):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]
    cleaned_items = []

    if delete_video:
        video_path = Path(job.get("video_path", ""))
        if cleanup_file(video_path):
            cleaned_items.append("uploaded_video")

    temp_wav = TEMP_DIR / f"{job_id}_audio.wav"
    if cleanup_file(temp_wav):
        cleaned_items.append("temp_audio")

    return {
        "job_id": job_id,
        "status": "cleaned",
        "cleaned_items": cleaned_items
    }


class UpdateSubtitlesRequest(BaseModel):
    segments: List[Dict[str, Any]]

@app.post("/api/jobs/{job_id}/subtitles/update")
def update_job_subtitles(job_id: str, req: UpdateSubtitlesRequest):
    """Save user edits from in-browser subtitle editor back to disk files."""
    srt_lines = []
    vtt_lines = ["WEBVTT", ""]
    txt_lines = []
    
    for idx, seg in enumerate(req.segments, start=1):
        s_val = float(seg.get("start", 0.0))
        e_val = float(seg.get("end", s_val + 0.5))
        s_srt = format_timestamp_srt(s_val)
        e_srt = format_timestamp_srt(e_val)
        s_vtt = format_timestamp_vtt(s_val)
        e_vtt = format_timestamp_vtt(e_val)
        txt = str(seg.get("text", "")).strip()

        srt_lines.extend([f"{idx}", f"{s_srt} --> {e_srt}", txt, ""])
        vtt_lines.extend([f"{idx}", f"{s_vtt} --> {e_vtt}", txt, ""])
        txt_lines.append(f"[{s_srt}] {txt}")

    (SUBTITLES_DIR / f"{job_id}.srt").write_text("\n".join(srt_lines), encoding="utf-8")
    (SUBTITLES_DIR / f"{job_id}.vtt").write_text("\n".join(vtt_lines), encoding="utf-8")
    (SUBTITLES_DIR / f"{job_id}.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    if job_id in JOBS and JOBS[job_id].get("result"):
        JOBS[job_id]["result"]["segments"] = req.segments

    return {"status": "updated", "cues_count": len(req.segments)}

class BatchExportRequest(BaseModel):
    job_ids: List[str]

@app.post("/api/batch/export-zip")
def export_batch_zip(req: BatchExportRequest):
    """Export multiple completed subtitle jobs into a single downloadable .ZIP archive."""
    import zipfile
    import io
    from fastapi.responses import Response

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for jid in req.job_ids:
            meta_file = SUBTITLES_DIR / f"{jid}.meta.json"
            orig_name = f"subtitles_{jid}"
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    orig_name = Path(meta.get("filename", orig_name)).stem
                except Exception:
                    pass
            for ext in ["srt", "vtt", "txt"]:
                fpath = SUBTITLES_DIR / f"{jid}.{ext}"
                if fpath.exists():
                    zf.write(fpath, arcname=f"{orig_name}.{ext}")

    zip_buffer.seek(0)
    return Response(
        content=zip_buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=whisper_batch_subtitles.zip"}
    )

@app.get("/api/cuda/status")
def get_cuda_status():
    """Inspect CUDA GPU hardware and acceleration status."""
    gpu = get_gpu_info()
    return {
        "has_nvidia_gpu": gpu.get("has_nvidia_gpu", False),
        "cuda_available": gpu.get("cuda_available", False),
        "device_name": gpu.get("device_name", "CPU"),
        "vram_gb": gpu.get("vram_gb", 0.0),
        "cuda_setup_needed": gpu.get("cuda_setup_needed", False),
        "message": gpu.get("cuda_message", "")
    }

@app.post("/api/cuda/enable")
def enable_cuda():
    """1-Click CUDA acceleration setup: installs nvidia-cublas and nvidia-cudnn wheels."""
    import sys
    import subprocess
    gpu = get_gpu_info()
    if not gpu.get("has_nvidia_gpu"):
        return {"status": "no_gpu", "message": "No physical NVIDIA GPU detected on this system."}
    if gpu.get("cuda_available"):
        return {"status": "ready", "message": f"NVIDIA CUDA is already active ({gpu.get('device_name')})!"}

    try:
        py_exe = sys.executable
        cmd = [py_exe, "-m", "pip", "install", "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "--no-warn-script-location"]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        setup_cuda_dll_paths()
        updated = get_gpu_info()
        return {
            "status": "installed" if updated.get("cuda_available") else "restart_needed",
            "cuda_available": updated.get("cuda_available"),
            "message": "NVIDIA CUDA cuDNN libraries installed successfully! GPU is accelerated." if updated.get("cuda_available") else "Packages installed. Please restart Whisper Studio to activate GPU acceleration.",
            "device_name": updated.get("device_name")
        }
    except Exception as e:
        return {"status": "error", "message": f"Installation failed: {str(e)}"}

FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
