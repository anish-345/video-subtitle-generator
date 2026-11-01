use axum::{
    body::Body,
    extract::{Path as AxumPath, Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
    routing::{delete, get, post},
    Router,
};
use axum::extract::Multipart;
use futures_util::StreamExt;
use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::Semaphore;
use tower_http::cors::CorsLayer;
use tower_http::services::ServeDir;

const WORKER_URL: &str = "http://127.0.0.1:8001";

fn which_cmd(cmd: &str) -> bool {
    #[cfg(windows)]
    let check = std::process::Command::new("where")
        .arg(cmd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    #[cfg(not(windows))]
    let check = std::process::Command::new("which")
        .arg(cmd)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    matches!(check, Ok(status) if status.success())
}

fn is_working_python(py: &Path) -> bool {
    let check = std::process::Command::new(py)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    matches!(check, Ok(status) if status.success())
}

fn find_python(base_dir: &Path) -> Option<PathBuf> {
    // 1. Embedded python in base_dir/python/python.exe
    let embedded = base_dir.join("python").join("python.exe");
    if embedded.is_file() && is_working_python(&embedded) {
        return Some(embedded);
    }
    // 2. Virtual environment in base_dir/.venv/Scripts/python.exe (or bin/python on Unix)
    #[cfg(windows)]
    let venv = base_dir.join(".venv").join("Scripts").join("python.exe");
    #[cfg(not(windows))]
    let venv = base_dir.join(".venv").join("bin").join("python");
    if venv.is_file() && is_working_python(&venv) {
        return Some(venv);
    }
    // 3. System python in PATH
    for cmd in ["python", "py", "python3"] {
        if which_cmd(cmd) {
            let p = PathBuf::from(cmd);
            if is_working_python(&p) {
                return Some(p);
            }
        }
    }
    None
}

#[derive(Clone)]
struct AppState {
    gpu_semaphore: Arc<Semaphore>,
    queue_counter: Arc<AtomicUsize>,
    http_client: reqwest::Client,
    #[allow(dead_code)]
    base_dir: PathBuf,
    uploads_dir: PathBuf,
    #[allow(dead_code)]
    subtitles_dir: PathBuf,
    temp_dir: PathBuf,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cur = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let base_dir = if cur.join("frontend").exists() {
        cur
    } else if let Some(p) = cur.parent() {
        if p.join("frontend").exists() {
            p.to_path_buf()
        } else {
            cur
        }
    } else {
        cur
    };
    let data_dir = base_dir.join("data");
    let uploads_dir = data_dir.join("uploads");
    let subtitles_dir = data_dir.join("subtitles");
    let temp_dir = data_dir.join("temp");

    for dir in [&uploads_dir, &subtitles_dir, &temp_dir] {
        let _ = fs::create_dir_all(dir).await;
    }

    let state = AppState {
        gpu_semaphore: Arc::new(Semaphore::new(1)), // Serializes GPU access to prevent CUDA OOM
        queue_counter: Arc::new(AtomicUsize::new(0)),
        http_client: reqwest::Client::builder()
            .timeout(Duration::from_secs(86400))
            .build()
            .unwrap(),
        base_dir: base_dir.clone(),
        uploads_dir,
        subtitles_dir,
        temp_dir,
    };

    // Background task: Periodic storage & memory manager
    let cleanup_state = state.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(900)).await;
            prune_expired_temp_files(&cleanup_state.temp_dir, 1800).await;
        }
    });

    let frontend_dir = base_dir.join("frontend");

    // Check if worker is responding; if not, auto-spawn it as child process
    let mut worker_child: Option<std::process::Child> = None;
    let worker_alive = state
        .http_client
        .get(format!("{WORKER_URL}/api/status"))
        .timeout(Duration::from_millis(1000))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if !worker_alive {
        if let Some(py_path) = find_python(&base_dir) {
            println!("🚀 Auto-starting GPU Whisper Worker on port 8001...");
            let mut cmd = std::process::Command::new(&py_path);
            cmd.args(["-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8001"]);
            cmd.current_dir(&base_dir);

            // Prepend base_dir/bin to PATH so local ffmpeg.exe is found
            let local_bin = base_dir.join("bin");
            if local_bin.exists() {
                if let Some(path_var) = std::env::var_os("PATH") {
                    let mut paths = std::env::split_paths(&path_var).collect::<Vec<_>>();
                    paths.insert(0, local_bin);
                    if let Ok(new_path) = std::env::join_paths(paths) {
                        cmd.env("PATH", new_path);
                    }
                }
            }

            match cmd.spawn() {
                Ok(mut child) => {
                    println!("✓ GPU Whisper Worker process spawned (PID {}). Waiting for port 8001...", child.id());
                    let mut ready = false;
                    for _ in 0..24 {
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        if let Ok(Some(status)) = child.try_wait() {
                            eprintln!("❌ Python worker exited prematurely with status: {status}");
                            break;
                        }
                        if state
                            .http_client
                            .get(format!("{WORKER_URL}/api/status"))
                            .timeout(Duration::from_millis(400))
                            .send()
                            .await
                            .map(|r| r.status().is_success())
                            .unwrap_or(false)
                        {
                            ready = true;
                            break;
                        }
                    }
                    if ready {
                        println!("✓ GPU Whisper Worker is healthy and listening on port 8001.");
                    } else {
                        eprintln!("⚠️ Python worker did not report ready in time. Transcriptions may fail if still initializing.");
                    }
                    worker_child = Some(child);
                }
                Err(e) => {
                    eprintln!("⚠️ Could not auto-spawn Python worker: {e}");
                }
            }
        } else {
            println!("⚠️ Python environment not found. Please ensure Python worker runs on port 8001.");
        }
    } else {
        println!("✓ Attached to existing GPU Worker on port 8001.");
    }

    // Auto-open browser in background
    tokio::spawn(async {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        #[cfg(windows)]
        let _ = std::process::Command::new("cmd").args(["/c", "start", "http://localhost:8000"]).spawn();
        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open").arg("http://localhost:8000").spawn();
        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open").arg("http://localhost:8000").spawn();
    });

    let app = Router::new()
        .route("/api/status", get(get_system_status))
        .route("/api/transcribe", post(handle_transcribe))
        .route("/api/jobs/:id/stream", get(proxy_job_stream))
        .route("/api/jobs/:id/status", get(proxy_job_status))
        .route("/api/jobs/:id/video", get(serve_job_video))
        .route("/api/jobs/:id/subtitles/:ext", get(serve_job_subtitles))
        .route("/api/jobs/:id/download/:ext", get(download_subtitles))
        .route("/api/jobs/:id/clean-audio", get(proxy_job_clean_audio))
        .route("/api/jobs/:id/cleanup", delete(cleanup_job))
        .route("/api/jobs/:id/subtitles/update", post(proxy_update_subtitles))
        .route("/api/cuda/status", get(proxy_cuda_status))
        .route("/api/cuda/enable", post(proxy_cuda_enable))
        .route("/api/batch/export-zip", post(proxy_batch_export_zip))
        .nest_service("/", ServeDir::new(frontend_dir))
        .layer(CorsLayer::permissive())
        .layer(axum::extract::DefaultBodyLimit::max(25 * 1024 * 1024 * 1024))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await.unwrap();
    println!("=================================================");
    println!("  🦀 WHISPER SUBTITLE STUDIO ACTIVE ON PORT 8000  ");
    println!("  UI & API: http://localhost:8000               ");
    println!("  Press Ctrl+C to stop the application.          ");
    println!("=================================================");

    let shutdown_signal = async {
        tokio::signal::ctrl_c().await.ok();
        println!("\n[Shutdown] Shutting down application...");
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await
        .unwrap();

    if let Some(mut child) = worker_child {
        println!("[Shutdown] Terminating GPU Worker child process...");
        let _ = child.kill();
        let _ = child.wait();
    }
}

async fn get_system_status(State(state): State<AppState>) -> impl IntoResponse {
    let queued = state.queue_counter.load(Ordering::Relaxed);
    
    match state.http_client.get(format!("{WORKER_URL}/api/status")).send().await {
        Ok(res) => {
            if let Ok(mut json) = res.json::<serde_json::Value>().await {
                if let Some(obj) = json.as_object_mut() {
                    let turbo_active = queued <= 1;
                    obj.insert(
                        "rust_core".to_string(),
                        serde_json::json!({
                            "engine": "Rust v1.92 + Tokio Async Gateway",
                            "status": "operational",
                            "queued_jobs": queued,
                            "gpu_concurrency_limit": 1,
                            "turbo_ready": turbo_active,
                            "acceleration_status": if turbo_active { "Maximum Power Turbo (170x Tensor Core)" } else { "Balanced Queue Serialization" },
                            "architecture": "Rust Orchestrator + Python CUDA Worker"
                        }),
                    );
                }
                return (StatusCode::OK, axum::Json(json)).into_response();
            }
        }
        Err(e) => {
            eprintln!("[Rust Core] Worker ping error: {e}");
        }
    }

    (
        StatusCode::OK,
        axum::Json(serde_json::json!({
            "status": "worker_initializing",
            "rust_core": {
                "engine": "Rust v1.92 + Tokio Async Gateway",
                "status": "running",
                "queued_jobs": queued
            }
        })),
    )
        .into_response()
}

async fn handle_transcribe(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let job_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
    let mut filename = "uploaded_video.mp4".to_string();
    let mut model_size = "base".to_string();
    let mut language = "auto".to_string();
    let mut task = "transcribe".to_string();
    let mut turbo_mode = true;
    let mut vad_filter = false;
    let mut noise_reduction = "studio".to_string();
    let mut saved_file_path: Option<PathBuf> = None;

    while let Some(mut field) = multipart.next_field().await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))? {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            if let Some(fname) = field.file_name() {
                filename = fname.to_string();
            }
            let sanitized: String = filename
                .chars()
                .filter(|c| c.is_alphanumeric() || matches!(*c, '.' | '_' | '-' | ' '))
                .collect();
            let dest_path = state.uploads_dir.join(format!("{job_id}_{sanitized}"));

            let mut file = tokio::fs::File::create(&dest_path).await.map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to create destination file: {e}"))
            })?;

            // Stream chunks directly from network to disk (zero RAM buffering)
            while let Some(chunk) = field.chunk().await.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))? {
                file.write_all(&chunk).await.map_err(|e| {
                    (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to write chunk: {e}"))
                })?;
            }
            file.flush().await.map_err(|e| {
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to flush file: {e}"))
            })?;

            saved_file_path = Some(dest_path);
        } else if name == "model_size" {
            if let Ok(txt) = field.text().await { model_size = txt; }
        } else if name == "language" {
            if let Ok(txt) = field.text().await { language = txt; }
        } else if name == "task" {
            if let Ok(txt) = field.text().await { task = txt; }
        } else if name == "turbo_mode" {
            if let Ok(txt) = field.text().await {
                turbo_mode = txt.trim().to_lowercase() != "false";
            }
        } else if name == "vad_filter" {
            if let Ok(txt) = field.text().await {
                vad_filter = txt.trim().to_lowercase() == "true";
            }
        } else if name == "noise_reduction" {
            if let Ok(txt) = field.text().await {
                noise_reduction = txt;
            }
        }
    }

    let Some(dest_path) = saved_file_path else {
        return Err((StatusCode::BAD_REQUEST, "No file provided".to_string()));
    };

    // BUG-09 Fix: Only count toward GPU queue after file upload has completed
    state.queue_counter.fetch_add(1, Ordering::SeqCst);
    let queue_counter = state.queue_counter.clone();

    // RAII guard to decrement queue counter when GPU processing finishes or fails
    struct QueueGuard(Arc<AtomicUsize>);
    impl Drop for QueueGuard {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let _guard = QueueGuard(queue_counter);

    // Acquire GPU Permit (Queue serialization - only 1 GPU job at a time to protect 6GB VRAM)
    let _permit = state.gpu_semaphore.acquire().await.map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("GPU Semaphore error: {e}"))
    })?;


    // Dynamic power management:
    // If queued <= 1 (solo job), unleash Maximum Power!
    // If queued > 1 (concurrency in flight), scale down batch size to conserve VRAM.
    let current_queued = state.queue_counter.load(Ordering::Relaxed);
    let effective_batch_size = if current_queued <= 1 {
        match model_size.as_str() {
            "tiny" | "base" => 16,
            "small" => 12,
            "medium" => 8,
            _ => 6,
        }
    } else {
        match model_size.as_str() {
            "tiny" | "base" => 8,
            "small" => 6,
            "medium" => 4,
            _ => 2,
        }
    };

    // Forward to GPU worker via local path handoff (zero duplicate copying)
    let payload = serde_json::json!({
        "job_id": job_id,
        "file_path": dest_path.to_string_lossy().to_string(),
        "filename": filename,
        "model_size": model_size,
        "language": language,
        "task": task,
        "turbo_mode": turbo_mode,
        "batch_size": effective_batch_size,
        "delete_video_after": false,
        "vad_filter": vad_filter,
        "noise_reduction": noise_reduction
    });

    let res = match state
        .http_client
        .post(format!("{WORKER_URL}/api/transcribe_path"))
        .json(&payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            let err_json = serde_json::json!({
                "detail": format!("GPU Worker connection failed (port 8001): {e}. Please ensure the Python worker is running.")
            });
            return Ok((
                StatusCode::BAD_GATEWAY,
                [(header::CONTENT_TYPE, "application/json")],
                err_json.to_string().into_bytes(),
            ));
        }
    };

    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
    let body_bytes = res.bytes().await.map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    Ok((status, [(header::CONTENT_TYPE, "application/json")], body_bytes.to_vec()))
}

async fn proxy_job_stream(
    AxumPath(job_id): AxumPath<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let url = format!("{WORKER_URL}/api/jobs/{job_id}/stream");
    match state.http_client.get(&url).send().await {
        Ok(res) => {
            let stream = res.bytes_stream().map(|result| {
                result.map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
            });
            let body = Body::from_stream(stream);
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
            headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            (StatusCode::OK, headers, body).into_response()
        }
        Err(e) => (StatusCode::BAD_GATEWAY, format!("Stream proxy error: {e}")).into_response(),
    }
}

async fn proxy_job_status(
    AxumPath(job_id): AxumPath<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let url = format!("{WORKER_URL}/api/jobs/{job_id}/status");
    match state.http_client.get(&url).send().await {
        Ok(res) => {
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
            let bytes = res.bytes().await.unwrap_or_default();
            (status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response()
        }
        Err(e) => {
            let err_json = serde_json::json!({
                "detail": format!("GPU Worker connection error: {e}")
            });
            (
                StatusCode::BAD_GATEWAY,
                [(header::CONTENT_TYPE, "application/json")],
                err_json.to_string().into_bytes(),
            )
                .into_response()
        }
    }
}

async fn proxy_cuda_status(State(state): State<AppState>) -> impl IntoResponse {
    let url = format!("{WORKER_URL}/api/cuda/status");
    match state.http_client.get(&url).send().await {
        Ok(res) => {
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
            let bytes = res.bytes().await.unwrap_or_default();
            (status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::json!({ "detail": format!("{e}") }).to_string().into_bytes(),
        ).into_response(),
    }
}

async fn proxy_cuda_enable(State(state): State<AppState>) -> impl IntoResponse {
    let url = format!("{WORKER_URL}/api/cuda/enable");
    match state.http_client.post(&url).send().await {
        Ok(res) => {
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
            let bytes = res.bytes().await.unwrap_or_default();
            (status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::json!({ "detail": format!("{e}") }).to_string().into_bytes(),
        ).into_response(),
    }
}

async fn proxy_update_subtitles(
    AxumPath(job_id): AxumPath<String>,
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let url = format!("{WORKER_URL}/api/jobs/{job_id}/subtitles/update");
    match state.http_client.post(&url).header("content-type", "application/json").body(body).send().await {
        Ok(res) => {
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
            let bytes = res.bytes().await.unwrap_or_default();
            (status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::json!({ "detail": format!("{e}") }).to_string().into_bytes(),
        ).into_response(),
    }
}

async fn proxy_batch_export_zip(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let url = format!("{WORKER_URL}/api/batch/export-zip");
    match state.http_client.post(&url).header("content-type", "application/json").body(body).send().await {
        Ok(res) => {
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
            let cd = if let Some(h) = res.headers().get(header::CONTENT_DISPOSITION) {
                h.clone()
            } else {
                HeaderValue::from_static("attachment; filename=\"whisper_batch_subtitles.zip\"")
            };
            let bytes = res.bytes().await.unwrap_or_default();
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/zip"));
            headers.insert(header::CONTENT_DISPOSITION, cd);
            (status, headers, bytes).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::json!({ "detail": format!("{e}") }).to_string().into_bytes(),
        ).into_response(),
    }
}

async fn serve_job_video(
    AxumPath(job_id): AxumPath<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let url = format!("{WORKER_URL}/api/jobs/{job_id}/video");
    match state.http_client.get(&url).send().await {
        Ok(res) => {
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
            let stream = res.bytes_stream().map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
            let body = Body::from_stream(stream);
            (status, [(header::CONTENT_TYPE, "video/mp4")], body).into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn serve_job_subtitles(
    AxumPath((job_id, ext)): AxumPath<(String, String)>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let url = format!("{WORKER_URL}/api/jobs/{job_id}/subtitles/{ext}");
    match state.http_client.get(&url).send().await {
        Ok(res) => {
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
            let ctype = if ext == "vtt" { "text/vtt" } else { "text/plain; charset=utf-8" };
            let bytes = res.bytes().await.unwrap_or_default();
            (status, [(header::CONTENT_TYPE, ctype)], bytes).into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn download_subtitles(
    AxumPath((job_id, ext)): AxumPath<(String, String)>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let url = format!("{WORKER_URL}/api/jobs/{job_id}/download/{ext}");
    match state.http_client.get(&url).send().await {
        Ok(res) => {
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
            let cd = if let Some(h) = res.headers().get(header::CONTENT_DISPOSITION) {
                h.clone()
            } else {
                HeaderValue::from_str(&format!("attachment; filename=\"subtitles_{job_id}.{ext}\""))
                    .unwrap_or_else(|_| HeaderValue::from_static("attachment"))
            };
            let bytes = res.bytes().await.unwrap_or_default();
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/octet-stream"));
            headers.insert(header::CONTENT_DISPOSITION, cd);
            (status, headers, bytes).into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn proxy_job_clean_audio(
    AxumPath(job_id): AxumPath<String>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let url = format!("{WORKER_URL}/api/jobs/{job_id}/clean-audio");
    match state.http_client.get(&url).send().await {
        Ok(res) => {
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
            let cd = res.headers().get(header::CONTENT_DISPOSITION).cloned();
            let stream = res.bytes_stream().map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e)));
            let body = Body::from_stream(stream);
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/mpeg"));
            if let Some(cd_val) = cd {
                headers.insert(header::CONTENT_DISPOSITION, cd_val);
            }
            (status, headers, body).into_response()
        }
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

#[derive(Deserialize)]
struct CleanupQuery {
    delete_video: Option<bool>,
}

async fn cleanup_job(
    AxumPath(job_id): AxumPath<String>,
    Query(query): Query<CleanupQuery>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let delete_video = query.delete_video.unwrap_or(true);
    let url = format!("{WORKER_URL}/api/jobs/{job_id}/cleanup?delete_video={delete_video}");
    match state.http_client.delete(&url).send().await {
        Ok(res) => {
            let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::OK);
            let bytes = res.bytes().await.unwrap_or_default();
            (status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response()
        }
        Err(e) => {
            let err_json = serde_json::json!({
                "detail": format!("GPU Worker connection error: {e}")
            });
            (
                StatusCode::BAD_GATEWAY,
                [(header::CONTENT_TYPE, "application/json")],
                err_json.to_string().into_bytes(),
            )
                .into_response()
        }
    }
}

async fn prune_expired_temp_files(dir: &Path, max_age_seconds: u64) {
    let Ok(mut entries) = fs::read_dir(dir).await else { return; };
    let now = std::time::SystemTime::now();

    while let Ok(Some(entry)) = entries.next_entry().await {
        if let Ok(metadata) = entry.metadata().await {
            if let Ok(modified) = metadata.modified() {
                if let Ok(elapsed) = now.duration_since(modified) {
                    if elapsed.as_secs() > max_age_seconds {
                        let _ = fs::remove_file(entry.path()).await;
                    }
                }
            }
        }
    }
}
