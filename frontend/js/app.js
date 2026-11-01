// Whisper Subtitle Studio - Advanced Batch Queue & Interactive Studio Controller

// Global App State
let batchQueue = []; // Array of { id, file, status, progress, jobId, result, error }
let activeBatchIndex = -1;
let isProcessingBatch = false;

let currentJobId = null;
let currentJobData = null;
let currentFile = null;
let eventSource = null;
let timerInterval = null;
let timerStart = null;
let parsedSegments = [];
let originalSegments = [];
let totalSyncOffset = 0.0;
let overlayEnabled = true;

// DOM Elements - Engine Badges
const rustBadge = document.getElementById("rustBadge");
const queueBadge = document.getElementById("queueBadge");
const gpuNameEl = document.getElementById("gpuName");
const gpuComputeTag = document.getElementById("gpuComputeTag");
const turboBadge = document.getElementById("turboBadge");
const turboBadgeText = document.getElementById("turboBadgeText");
const turboToggle = document.getElementById("turboToggle");
const queueStatusPill = document.getElementById("queueStatusPill");
const speedupPill = document.getElementById("speedupPill");
const batchProgressPill = document.getElementById("batchProgressPill");

// CUDA 1-Click Notice Banner
const cudaNoticeBanner = document.getElementById("cudaNoticeBanner");
const cudaNoticeTitle = document.getElementById("cudaNoticeTitle");
const cudaNoticeText = document.getElementById("cudaNoticeText");
const installCudaBtn = document.getElementById("installCudaBtn");
const installCudaBtnText = document.getElementById("installCudaBtnText");

// Upload & Setup Form
const dropzone = document.getElementById("dropzone");
const videoFileInput = document.getElementById("videoFileInput");
const dropzonePrompt = document.getElementById("dropzonePrompt");

const batchQueueContainer = document.getElementById("batchQueueContainer");
const batchCountBadge = document.getElementById("batchCountBadge");
const batchSizeHint = document.getElementById("batchSizeHint");
const batchFileList = document.getElementById("batchFileList");
const clearBatchBtn = document.getElementById("clearBatchBtn");
const addMoreFilesBtn = document.getElementById("addMoreFilesBtn");
const downloadBatchZipBtn = document.getElementById("downloadBatchZipBtn");
const batchStatusSummary = document.getElementById("batchStatusSummary");

const modelSelect = document.getElementById("modelSelect");
const languageSelect = document.getElementById("languageSelect");
const taskSelect = document.getElementById("taskSelect");
const coverageSelect = document.getElementById("coverageSelect");
const noiseReductionSelect = document.getElementById("noiseReductionSelect");
const generateBtn = document.getElementById("generateBtn");
const generateBtnText = document.getElementById("generateBtnText");

// Progress & Live Stream
const setupSection = document.getElementById("setupSection");
const progressSection = document.getElementById("progressSection");
const progressBar = document.getElementById("progressBar");
const progressPercent = document.getElementById("progressPercent");
const statusMessage = document.getElementById("statusMessage");
const elapsedTimer = document.getElementById("elapsedTimer");
const accurateProgressPill = document.getElementById("accurateProgressPill");
const etaTimerPill = document.getElementById("etaTimerPill");
const logTerminal = document.getElementById("logTerminal");
const cancelJobBtn = document.getElementById("cancelJobBtn");

// Subtitle Studio & Video Player Elements
const resultsSection = document.getElementById("resultsSection");
const statLang = document.getElementById("statLang");
const statCues = document.getElementById("statCues");
const statWords = document.getElementById("statWords");
const statDuration = document.getElementById("statDuration");
const statSpeed = document.getElementById("statSpeed");
const statNoise = document.getElementById("statNoise");

const downloadSrtBtn = document.getElementById("downloadSrtBtn");
const downloadVttBtn = document.getElementById("downloadVttBtn");
const downloadTxtBtn = document.getElementById("downloadTxtBtn");
const downloadCleanAudioBtn = document.getElementById("downloadCleanAudioBtn");
const saveEditsBtn = document.getElementById("saveEditsBtn");

const studioVideoPlayer = document.getElementById("studioVideoPlayer");
const videoSubtitleOverlay = document.getElementById("videoSubtitleOverlay");
const overlaySubtitleText = document.getElementById("overlaySubtitleText");
const videoSkipBackBtn = document.getElementById("videoSkipBackBtn");
const videoPlayPauseBtn = document.getElementById("videoPlayPauseBtn");
const videoSkipFwdBtn = document.getElementById("videoSkipFwdBtn");
const videoSpeedSelect = document.getElementById("videoSpeedSelect");
const toggleSubtitleOverlayBtn = document.getElementById("toggleSubtitleOverlayBtn");

const subtitleSearchInput = document.getElementById("subtitleSearchInput");
const searchMatchCount = document.getElementById("searchMatchCount");
const shiftBackBtn = document.getElementById("shiftBackBtn");
const shiftFwdBtn = document.getElementById("shiftFwdBtn");
const offsetDisplay = document.getElementById("offsetDisplay");
const resetShiftBtn = document.getElementById("resetShiftBtn");
const addCueBtn = document.getElementById("addCueBtn");
const copyAllBtn = document.getElementById("copyAllBtn");
const saveStatusBadge = document.getElementById("saveStatusBadge");
const subtitleTableBody = document.getElementById("subtitleTableBody");
const editorTableContainer = document.getElementById("editorTableContainer");

// =========================================================================
// 1. Initial Status Check & CUDA Check
// =========================================================================
async function initSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("Failed to query status");
    const data = await res.json();

    // Wi-Fi Network Badge
    const wifiBadge = document.getElementById("wifiBadge");
    const wifiName = document.getElementById("wifiName");
    const wifiBadgeCopy = document.getElementById("wifiBadgeCopy");
    if (wifiBadge && data.wifi_ip) {
      const port = window.location.port || "8000";
      const fullWifiUrl = `http://${data.wifi_ip}:${port}`;
      if (wifiName) wifiName.textContent = `📶 Wi-Fi: ${data.wifi_ip}:${port}`;
      wifiBadge.onclick = () => {
        navigator.clipboard.writeText(fullWifiUrl).then(() => {
          if (wifiBadgeCopy) {
            wifiBadgeCopy.textContent = "Copied!";
            wifiBadgeCopy.style.background = "#bbf7d0";
            wifiBadgeCopy.style.color = "#15803d";
            setTimeout(() => {
              wifiBadgeCopy.textContent = "Copy";
              wifiBadgeCopy.style.background = "#dbeafe";
              wifiBadgeCopy.style.color = "#1d4ed8";
            }, 2000);
          }
        });
      };
    }

    // Rust Core Metrics
    if (data.rust_core) {
      const q = data.rust_core.queued_jobs || 0;
      if (queueBadge) queueBadge.textContent = `${q} in Queue`;
      if (turboBadgeText) {
        turboBadgeText.textContent = data.rust_core.turbo_ready ? "170x Speed (10h in ~4m)" : "Queue Balanced";
      }
    }

    // GPU Metrics
    if (data.gpu && data.gpu.cuda_available) {
      if (gpuNameEl) gpuNameEl.textContent = `${data.gpu.device_name} (${data.gpu.vram_gb} GB)`;
      if (gpuComputeTag) gpuComputeTag.textContent = data.gpu.recommended_compute.toUpperCase();
    } else {
      if (gpuNameEl) gpuNameEl.textContent = data.gpu && data.gpu.has_nvidia_gpu ? `${data.gpu.device_name} (CPU mode)` : "CPU Fallback Mode";
      if (gpuComputeTag) gpuComputeTag.textContent = "CPU";
    }

    // Check CUDA status for 1-click banner
    checkCudaStatus();

    // Update models if provided by backend
    if (data.available_models && Array.isArray(data.available_models) && modelSelect) {
      updateModelOptions(data.available_models);
    }
  } catch (err) {
    console.warn("Status check notice:", err);
  }
}

async function checkCudaStatus() {
  try {
    const res = await fetch("/api/cuda/status");
    if (!res.ok) return;
    const data = await res.json();

    if (data.has_nvidia_gpu && data.cuda_setup_needed && !data.cuda_available) {
      if (cudaNoticeBanner) {
        cudaNoticeBanner.classList.remove("hidden");
        if (cudaNoticeTitle) cudaNoticeTitle.textContent = `⚡ ${data.device_name} Detected!`;
        if (cudaNoticeText) cudaNoticeText.textContent = "An NVIDIA GPU is present, but cuDNN acceleration libraries are missing. Click to enable 1-Click CUDA Acceleration (20x-50x speedup).";
      }
    } else {
      if (cudaNoticeBanner) cudaNoticeBanner.classList.add("hidden");
    }
  } catch (e) {
    console.warn("CUDA check error:", e);
  }
}

// 1-Click CUDA Enable Handler
if (installCudaBtn) {
  installCudaBtn.addEventListener("click", async () => {
    installCudaBtn.disabled = true;
    if (installCudaBtnText) installCudaBtnText.textContent = "⏳ Installing CUDA & cuDNN (please wait ~30-60s)...";
    try {
      const res = await fetch("/api/cuda/enable", { method: "POST" });
      const data = await res.json();
      if (data.status === "installed" || data.cuda_available) {
        alert(data.message || "CUDA Acceleration Enabled successfully!");
        if (cudaNoticeBanner) cudaNoticeBanner.classList.add("hidden");
        initSystemStatus();
      } else {
        alert(data.message || "Installation finished. Please restart the app to activate CUDA.");
      }
    } catch (err) {
      alert("CUDA setup notice: " + err.message);
    } finally {
      installCudaBtn.disabled = false;
      if (installCudaBtnText) installCudaBtnText.textContent = "⚡ Enable 1-Click CUDA Acceleration";
    }
  });
}

function updateModelOptions(models) {
  const currentVal = modelSelect.value;
  modelSelect.innerHTML = "";
  models.forEach(m => {
    const opt = document.createElement("option");
    opt.value = m.id;
    let tag = m.cached ? "[Instant ⚡]" : "[Download]";
    if (m.id === "base") tag = "[⚡ 170x Turbo • Instant Start - Recommended]";
    else if (m.id === "sensevoice-small") tag = "[✨ Millisecond Precision • SOTA Emotion & Speech • Instant Start]";
    else if (m.id === "large-v3-turbo") tag = "[🚀 Deep Learning SOTA • Instant Start]";
    else if (m.id === "medium") tag = "[🎯 Studio Quality • Instant Start]";
    else if (m.id === "small") tag = "[✨ High Accuracy • Instant Start]";
    else if (m.id === "tiny") tag = "[⚡ Ultra Fast • Instant Start]";
    else if (m.id === "large-v3") tag = m.cached ? "[🧠 Multilingual Max • Instant Start]" : "[🧠 Multilingual Max (~3GB)]";
    opt.textContent = `${m.name} ${tag}`;
    if (m.id === currentVal || (m.recommended && !currentVal)) {
      opt.selected = true;
    }
    modelSelect.appendChild(opt);
  });
}

// =========================================================================
// 2. Drag & Drop & Multi-File Batch Queue Setup
// =========================================================================
function setupDropzone() {
  if (!dropzone) return;

  ["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    }, false);
  });

  ["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    }, false);
  });

  dropzone.addEventListener("drop", (e) => {
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length > 0) {
      addFilesToQueue(files);
    }
  });

  if (videoFileInput) {
    videoFileInput.addEventListener("change", (e) => {
      const files = Array.from(e.target.files || []);
      if (files.length > 0) {
        addFilesToQueue(files);
      }
    });
  }

  if (addMoreFilesBtn) {
    addMoreFilesBtn.addEventListener("click", () => {
      if (videoFileInput) videoFileInput.click();
    });
  }

  if (clearBatchBtn) {
    clearBatchBtn.addEventListener("click", () => {
      if (isProcessingBatch && !confirm("A batch queue is in progress. Are you sure you want to clear?")) return;
      resetBatchQueue();
    });
  }

  if (downloadBatchZipBtn) {
    downloadBatchZipBtn.addEventListener("click", downloadBatchZip);
  }
}

function addFilesToQueue(newFiles) {
  let addedCount = 0;
  newFiles.forEach(file => {
    if (file.size > 25 * 1024 * 1024 * 1024) {
      alert(`File "${file.name}" exceeds the 25 GB maximum limit.`);
      return;
    }
    // Avoid duplicate files by name and size
    const exists = batchQueue.some(item => item.file.name === file.name && item.file.size === file.size);
    if (!exists) {
      batchQueue.push({
        id: "q_" + Math.random().toString(36).substring(2, 9),
        file: file,
        status: "queued",
        progress: 0,
        jobId: null,
        result: null,
        error: null
      });
      addedCount++;
    }
  });

  if (addedCount > 0) {
    currentFile = batchQueue[0].file;
    renderBatchQueueUI();
  }
}

function removeFileFromQueue(index) {
  if (isProcessingBatch && activeBatchIndex === index) {
    alert("Cannot remove a file currently being processed.");
    return;
  }
  batchQueue.splice(index, 1);
  if (batchQueue.length === 0) {
    resetBatchQueue();
  } else {
    currentFile = batchQueue[0].file;
    renderBatchQueueUI();
  }
}

function resetBatchQueue() {
  batchQueue = [];
  activeBatchIndex = -1;
  isProcessingBatch = false;
  currentFile = null;
  if (videoFileInput) videoFileInput.value = "";
  if (batchQueueContainer) batchQueueContainer.classList.add("hidden");
  if (generateBtn) {
    generateBtn.disabled = true;
    if (generateBtnText) generateBtnText.textContent = "Generate Subtitles (.SRT)";
  }
}

function renderBatchQueueUI() {
  if (!batchQueueContainer || !batchFileList) return;

  if (batchQueue.length === 0) {
    batchQueueContainer.classList.add("hidden");
    if (generateBtn) generateBtn.disabled = true;
    return;
  }

  batchQueueContainer.classList.remove("hidden");

  let totalBytes = 0;
  let completedCount = 0;

  batchFileList.innerHTML = "";
  batchQueue.forEach((item, idx) => {
    totalBytes += item.file.size;
    if (item.status === "completed") completedCount++;

    const card = document.createElement("div");
    card.className = `batch-file-card ${item.status}`;
    card.id = `item-${item.id}`;

    let statusPillHtml = '<span class="badge">Queued</span>';
    if (item.status === "processing") {
      statusPillHtml = `<span class="badge badge-turbo">⚡ Transcribing (${item.progress}%)</span>`;
    } else if (item.status === "completed") {
      statusPillHtml = '<span class="badge badge-emerald">✓ Completed</span>';
    } else if (item.status === "failed") {
      statusPillHtml = '<span class="badge" style="background:#fee2e2; color:#b91c1c;">✕ Error</span>';
    }

    let actionBtnHtml = `
      <button type="button" class="btn-icon-sm delete" title="Remove from queue" onclick="removeFileFromQueue(${idx})">✕</button>
    `;
    if (item.status === "completed" && item.result) {
      actionBtnHtml = `
        <button type="button" class="btn btn-secondary btn-sm" onclick="loadCompletedJobToStudio(${idx})">Open in Studio</button>
        ${actionBtnHtml}
      `;
    }

    card.innerHTML = `
      <div class="batch-file-left">
        <span class="batch-file-icon">🎬</span>
        <div>
          <div class="batch-file-name" title="${escapeHtml(item.file.name)}">${escapeHtml(item.file.name)}</div>
          <div class="batch-file-size">${formatBytes(item.file.size)}</div>
        </div>
      </div>
      <div class="batch-file-right">
        ${statusPillHtml}
        ${actionBtnHtml}
      </div>
    `;
    batchFileList.appendChild(card);
  });

  if (batchCountBadge) batchCountBadge.textContent = `${batchQueue.length} file${batchQueue.length > 1 ? "s" : ""}`;
  if (batchSizeHint) batchSizeHint.textContent = `Total: ${formatBytes(totalBytes)}`;

  if (generateBtn) {
    generateBtn.disabled = isProcessingBatch;
    if (generateBtnText) {
      if (batchQueue.length > 1) {
        generateBtnText.textContent = `⚡ Process Batch (${batchQueue.length} Videos)`;
      } else {
        generateBtnText.textContent = "Generate Subtitles (.SRT)";
      }
    }
  }

  // Show Download All Zip button if any files are completed
  if (downloadBatchZipBtn) {
    if (completedCount > 1) {
      downloadBatchZipBtn.classList.remove("hidden");
      downloadBatchZipBtn.textContent = `📦 Download All Subtitles (${completedCount} .ZIP)`;
    } else {
      downloadBatchZipBtn.classList.add("hidden");
    }
  }

  if (batchStatusSummary) {
    if (isProcessingBatch) {
      batchStatusSummary.textContent = `Processing video ${activeBatchIndex + 1} of ${batchQueue.length}...`;
    } else if (completedCount === batchQueue.length && batchQueue.length > 0) {
      batchStatusSummary.textContent = `✓ All ${batchQueue.length} files transcribed successfully!`;
    } else {
      batchStatusSummary.textContent = "Sequential batch queue ready";
    }
  }
}

window.removeFileFromQueue = removeFileFromQueue;
window.loadCompletedJobToStudio = (index) => {
  const item = batchQueue[index];
  if (item && item.result) {
    currentFile = item.file;
    currentJobId = item.jobId;
    finishJob(item.result);
  }
};

function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
  const i = Math.min(sizes.length - 1, Math.floor(Math.log(bytes) / Math.log(k)));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// =========================================================================
// 3. Batch Queue Sequential Executor
// =========================================================================
if (generateBtn) {
  generateBtn.addEventListener("click", async () => {
    if (batchQueue.length === 0 || isProcessingBatch) return;

    isProcessingBatch = true;
    generateBtn.disabled = true;

    // Process from the first non-completed item
    for (let i = 0; i < batchQueue.length; i++) {
      if (batchQueue[i].status === "completed") continue;

      activeBatchIndex = i;
      const currentItem = batchQueue[i];
      currentItem.status = "processing";
      renderBatchQueueUI();

      if (batchProgressPill) {
        batchProgressPill.textContent = `Video ${i + 1} of ${batchQueue.length}`;
        batchProgressPill.classList.remove("hidden");
      }

      const success = await processSingleItem(currentItem);
      if (success) {
        currentItem.status = "completed";
      } else {
        currentItem.status = "failed";
      }
      renderBatchQueueUI();
    }

    isProcessingBatch = false;
    generateBtn.disabled = false;
    renderBatchQueueUI();
  });
}

async function processSingleItem(item) {
  currentFile = item.file;
  if (progressSection) progressSection.classList.remove("hidden");
  if (queueStatusPill) queueStatusPill.classList.add("hidden");
  if (speedupPill) speedupPill.classList.add("hidden");
  if (logTerminal) logTerminal.innerHTML = "";
  if (progressBar) progressBar.style.width = "2%";
  if (progressPercent) progressPercent.textContent = "0%";
  if (accurateProgressPill) accurateProgressPill.textContent = "00:00 / 00:00 (0%)";
  if (etaTimerPill) etaTimerPill.textContent = "ETA: Calculating...";
  if (statusMessage) statusMessage.textContent = `Uploading "${item.file.name}" to Rust Gateway...`;

  startTimer();

  const isTurbo = turboToggle ? turboToggle.checked : true;
  const useVad = coverageSelect ? coverageSelect.value === "vad" : false;
  const noiseVal = noiseReductionSelect ? noiseReductionSelect.value : "studio";

  addLog(`[Rust Core] Submitting "${item.file.name}" (${formatBytes(item.file.size)})...`);
  addLog(`[Config] Model: ${modelSelect.value} | Language: ${languageSelect.value} | Noise: ${noiseVal} | Turbo: ${isTurbo ? '170x' : 'Balanced'}`);

  const formData = new FormData();
  formData.append("file", item.file);
  formData.append("model_size", modelSelect.value);
  formData.append("language", languageSelect.value);
  formData.append("task", taskSelect.value);
  formData.append("turbo_mode", isTurbo ? "true" : "false");
  formData.append("vad_filter", useVad ? "true" : "false");
  formData.append("noise_reduction", noiseVal);

  try {
    const res = await fetch("/api/transcribe", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      let errMsg = "Upload request failed";
      try {
        const err = await res.json();
        errMsg = err.detail || err.message || JSON.stringify(err);
      } catch {
        errMsg = await res.text();
      }
      throw new Error(errMsg);
    }

    const data = await res.json();
    currentJobId = data.job_id;
    item.jobId = data.job_id;

    if (data.status === "completed" && data.result) {
      stopTimer();
      item.result = data.result;
      finishJob(data.result);
      return true;
    }

    addLog(`[Rust Gateway] Job assigned: ${currentJobId}. Subscribed to live GPU event stream...`, "success");

    // Await completion via Promise wrapped around EventSource / Polling
    const result = await waitForJobCompletion(currentJobId, item);
    item.result = result;
    finishJob(result);
    return true;
  } catch (err) {
    stopTimer();
    item.error = err.message;
    if (statusMessage) statusMessage.textContent = "Error: " + err.message;
    addLog("Error: " + err.message, "error");
    return false;
  }
}

function waitForJobCompletion(jobId, item) {
  return new Promise((resolve, reject) => {
    let sse = new EventSource(`/api/jobs/${jobId}/stream`);

    sse.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        handleProgressUpdate(payload, item);

        if (payload.step === "completed") {
          sse.close();
          stopTimer();
          fetch(`/api/jobs/${jobId}/status`)
            .then(r => r.json())
            .then(d => resolve(d.result))
            .catch(reject);
        } else if (payload.step === "error") {
          sse.close();
          stopTimer();
          reject(new Error(payload.message || "Transcription failed"));
        }
      } catch (e) {
        console.warn("SSE parse error", e);
      }
    };

    sse.onerror = async () => {
      sse.close();
      // Poll fallback
      const result = await pollJobStatusUntilDone(jobId, item);
      if (result) resolve(result);
      else reject(new Error("Polling timed out or job failed"));
    };
  });
}

async function pollJobStatusUntilDone(jobId, item, maxAttempts = 120, intervalMs = 2500) {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, intervalMs));
    try {
      const res = await fetch(`/api/jobs/${jobId}/status`);
      if (!res.ok) continue;
      const data = await res.json();

      if (data.status === "completed" && data.result) {
        stopTimer();
        return data.result;
      } else if (data.status === "failed") {
        stopTimer();
        throw new Error(data.error || "Job failed");
      } else if (data.status === "processing" || data.status === "queued") {
        handleProgressUpdate({ progress: data.progress, message: data.message }, item);
      }
    } catch (e) {
      console.warn("Poll attempt failed, retrying...", e);
    }
  }
  return null;
}

// =========================================================================
// 4. Accurate Percentage Progress & Live Status Handling
// =========================================================================
function handleProgressUpdate(data, currentQueueItem) {
  if (data.progress !== undefined) {
    const pct = Math.min(100, Math.max(0, data.progress));
    if (progressBar) progressBar.style.width = `${pct}%`;
    if (progressPercent) progressPercent.textContent = `${pct}%`;
    if (currentQueueItem) currentQueueItem.progress = pct;
  }

  // Exact time progress (e.g. 02:15 / 05:00 (45%))
  if (data.current_time_fmt && data.total_duration_fmt && accurateProgressPill) {
    accurateProgressPill.textContent = `${data.current_time_fmt} / ${data.total_duration_fmt} (${data.progress}%)`;
    accurateProgressPill.classList.remove("hidden");
  }

  // Estimated Time Remaining (ETA)
  if (data.eta_fmt && etaTimerPill) {
    etaTimerPill.textContent = `ETA: ~${data.eta_fmt}`;
    etaTimerPill.classList.remove("hidden");
  }

  if (data.message) {
    if (statusMessage) statusMessage.textContent = data.message;
    addLog(data.message);
  }

  if (data.queue_position !== undefined && data.queue_position > 0 && queueStatusPill) {
    queueStatusPill.textContent = `Queue Position #${data.queue_position}`;
    queueStatusPill.classList.remove("hidden");
  } else if (queueStatusPill) {
    queueStatusPill.classList.add("hidden");
  }

  if (data.speedup_factor && data.speedup_factor > 1 && speedupPill) {
    speedupPill.textContent = `⚡ ${data.speedup_factor}x Speedup`;
    speedupPill.classList.remove("hidden");
  }
}

// =========================================================================
// 5. Studio Results & Synchronized Video Player Preview
// =========================================================================
function finishJob(result) {
  currentJobData = result;
  originalSegments = JSON.parse(JSON.stringify(result.segments || []));
  parsedSegments = JSON.parse(JSON.stringify(result.segments || []));
  totalSyncOffset = 0.0;
  if (offsetDisplay) offsetDisplay.textContent = "0.0s";

  if (resultsSection) resultsSection.classList.remove("hidden");

  // Metrics
  const langName = (result.detected_language || "en").toUpperCase();
  const cueCount = parsedSegments.length;
  let wordCount = 0;
  parsedSegments.forEach(s => {
    wordCount += s.text.trim().split(/\s+/).filter(w => w.length > 0).length;
  });

  const durationSec = result.total_duration || (parsedSegments.length > 0 ? parsedSegments[parsedSegments.length - 1].end : 0);
  const durMin = Math.floor(durationSec / 60);
  const durSec = Math.floor(durationSec % 60);
  const durFormatted = durMin > 0 ? `${durMin}m ${durSec}s` : `${durationSec.toFixed(1)}s`;

  if (statLang) statLang.textContent = `🌐 ${langName} (${result.language_probability ? result.language_probability + '%' : '100%'})`;
  if (statCues) statCues.textContent = `📝 ${cueCount} Cues`;
  if (statWords) statWords.textContent = `💬 ${wordCount} Words`;
  if (statDuration) statDuration.textContent = `⏱️ ${durFormatted}`;
  if (statSpeed) statSpeed.textContent = `⚡ ${result.speedup_factor || 170}x Acceleration`;

  if (statNoise) {
    const nMap = {
      "studio": "🎙️ Denoised: Studio Clarity",
      "aggressive": "🌪️ Denoised: Aggressive",
      "vocal_boost": "🔊 Denoised: Voice Boost",
      "off": "⚪ Denoised: Off"
    };
    statNoise.textContent = nMap[result.noise_reduction] || "🎙️ Denoised: Studio Clarity";
  }

  // Setup Download Links
  const originalName = result.filename || (currentFile ? currentFile.name : `subtitles_${result.job_id}`);
  const stem = originalName.replace(/\.[^/.]+$/, "");

  if (downloadSrtBtn) {
    downloadSrtBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-svg">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
      Download ${escapeHtml(stem)}.srt
    `;
    downloadSrtBtn.onclick = () => window.location.href = `/api/jobs/${result.job_id}/download/srt`;
  }

  if (downloadVttBtn) {
    downloadVttBtn.textContent = `${stem}.vtt`;
    downloadVttBtn.onclick = () => window.location.href = `/api/jobs/${result.job_id}/download/vtt`;
  }

  if (downloadTxtBtn) {
    downloadTxtBtn.textContent = `${stem}.txt`;
    downloadTxtBtn.onclick = () => window.location.href = `/api/jobs/${result.job_id}/download/txt`;
  }

  if (downloadCleanAudioBtn) {
    if (result.clean_audio_url) {
      downloadCleanAudioBtn.classList.remove("hidden");
      downloadCleanAudioBtn.onclick = () => window.location.href = `/api/jobs/${result.job_id}/clean-audio`;
    } else {
      downloadCleanAudioBtn.classList.add("hidden");
    }
  }

  // Bind HTML5 Video Player
  setupVideoPlayer(result);

  // Render Interactive Subtitle Table
  renderSubtitleTable(parsedSegments);

  resultsSection.scrollIntoView({ behavior: "smooth" });
}

// =========================================================================
// 6. Video Player Synchronization & Controls
// =========================================================================
function setupVideoPlayer(result) {
  if (!studioVideoPlayer) return;

  // Use local in-memory Object URL if user uploaded file, else load server endpoint
  if (currentFile && currentFile instanceof File) {
    try {
      studioVideoPlayer.src = URL.createObjectURL(currentFile);
    } catch {
      studioVideoPlayer.src = `/api/jobs/${result.job_id}/video`;
    }
  } else {
    studioVideoPlayer.src = `/api/jobs/${result.job_id}/video`;
  }

  studioVideoPlayer.load();

  // Synchronized Subtitle Overlay update on timeupdate
  studioVideoPlayer.ontimeupdate = () => {
    const curTime = studioVideoPlayer.currentTime;
    syncSubtitleOverlayAndTable(curTime);
  };

  if (videoPlayPauseBtn) {
    videoPlayPauseBtn.onclick = () => {
      if (studioVideoPlayer.paused) {
        studioVideoPlayer.play();
        videoPlayPauseBtn.textContent = "⏸️ Pause";
      } else {
        studioVideoPlayer.pause();
        videoPlayPauseBtn.textContent = "⏯️ Play";
      }
    };
  }

  if (videoSkipBackBtn) {
    videoSkipBackBtn.onclick = () => {
      studioVideoPlayer.currentTime = Math.max(0, studioVideoPlayer.currentTime - 5);
    };
  }

  if (videoSkipFwdBtn) {
    videoSkipFwdBtn.onclick = () => {
      studioVideoPlayer.currentTime = Math.min(studioVideoPlayer.duration || Infinity, studioVideoPlayer.currentTime + 5);
    };
  }

  if (videoSpeedSelect) {
    videoSpeedSelect.onchange = (e) => {
      studioVideoPlayer.playbackRate = parseFloat(e.target.value) || 1.0;
    };
  }

  if (toggleSubtitleOverlayBtn) {
    toggleSubtitleOverlayBtn.onclick = () => {
      overlayEnabled = !overlayEnabled;
      if (videoSubtitleOverlay) {
        videoSubtitleOverlay.style.display = overlayEnabled ? "block" : "none";
      }
      toggleSubtitleOverlayBtn.textContent = overlayEnabled ? "💬 Overlay ON" : "💬 Overlay OFF";
      toggleSubtitleOverlayBtn.classList.toggle("active", overlayEnabled);
    };
  }
}

function syncSubtitleOverlayAndTable(currentTime) {
  let activeCue = null;
  let activeIndex = -1;

  for (let i = 0; i < parsedSegments.length; i++) {
    const seg = parsedSegments[i];
    if (currentTime >= seg.start && currentTime <= seg.end) {
      activeCue = seg;
      activeIndex = i;
      break;
    }
  }

  // Update on-screen overlay
  if (videoSubtitleOverlay && overlaySubtitleText) {
    if (activeCue && overlayEnabled) {
      overlaySubtitleText.textContent = activeCue.text;
      videoSubtitleOverlay.classList.remove("hidden");
    } else {
      videoSubtitleOverlay.classList.add("hidden");
    }
  }

  // Highlight and smoothly scroll table row
  const rows = subtitleTableBody ? subtitleTableBody.querySelectorAll(".subtitle-row") : [];
  rows.forEach((row, idx) => {
    if (idx === activeIndex) {
      if (!row.classList.contains("active")) {
        row.classList.add("active");
        // Scroll only if row is not visible
        const container = editorTableContainer;
        if (container) {
          const rowTop = row.offsetTop;
          const rowBottom = rowTop + row.offsetHeight;
          const viewTop = container.scrollTop;
          const viewBottom = viewTop + container.clientHeight;
          if (rowTop < viewTop || rowBottom > viewBottom) {
            row.scrollIntoView({ behavior: "smooth", block: "nearest" });
          }
        }
      }
    } else {
      row.classList.remove("active");
    }
  });
}

// =========================================================================
// 7. Interactive Subtitle Table & Inline Editor
// =========================================================================
function renderSubtitleTable(segments, filterText = "") {
  if (!subtitleTableBody) return;
  subtitleTableBody.innerHTML = "";

  if (!segments || segments.length === 0) {
    subtitleTableBody.innerHTML = '<tr><td colspan="4" class="hint" style="text-align:center; padding: 32px;">No speech segments found.</td></tr>';
    return;
  }

  const query = filterText.toLowerCase().trim();
  let matchCount = 0;

  segments.forEach((seg, idx) => {
    const textLower = (seg.text || "").toLowerCase();
    const timeMatch = (seg.start_fmt || "").includes(query) || (seg.end_fmt || "").includes(query);
    const textMatch = textLower.includes(query);
    const isVisible = query === "" || textMatch || timeMatch;

    if (isVisible) matchCount++;

    const row = document.createElement("tr");
    row.className = `subtitle-row ${isVisible ? "" : "hidden"}`;
    row.id = `row-${idx}`;
    row.dataset.index = idx;

    row.innerHTML = `
      <td class="row-index">${idx + 1}</td>
      <td class="timing-cell">
        <div class="timing-inputs">
          <button type="button" class="timing-pill-btn" onclick="seekVideoTo(${seg.start})" title="Jump video to start time">
            ${seg.start_fmt || formatSecondsToSrt(seg.start)}
          </button>
          <span class="timing-arrow">➔</span>
          <button type="button" class="timing-pill-btn" onclick="seekVideoTo(${seg.end})" title="Jump video to end time">
            ${seg.end_fmt || formatSecondsToSrt(seg.end)}
          </button>
        </div>
      </td>
      <td class="text-cell">
        <textarea class="subtitle-edit-input" oninput="updateSegmentText(${idx}, this.value)" rows="1">${escapeHtml(seg.text)}</textarea>
      </td>
      <td class="action-cell">
        <div class="row-actions">
          <button type="button" class="btn-icon-sm" onclick="playSegment(${idx})" title="Play this segment">▶</button>
          <button type="button" class="btn-icon-sm delete" onclick="deleteSegment(${idx})" title="Delete subtitle line">🗑️</button>
        </div>
      </td>
    `;
    subtitleTableBody.appendChild(row);
  });

  if (searchMatchCount) {
    if (query !== "") {
      searchMatchCount.textContent = `${matchCount} matches`;
      searchMatchCount.classList.remove("hidden");
    } else {
      searchMatchCount.classList.add("hidden");
    }
  }
}

window.seekVideoTo = (seconds) => {
  if (studioVideoPlayer) {
    studioVideoPlayer.currentTime = Math.max(0, seconds);
    studioVideoPlayer.play();
    if (videoPlayPauseBtn) videoPlayPauseBtn.textContent = "⏸️ Pause";
  }
};

window.playSegment = (index) => {
  const seg = parsedSegments[index];
  if (!seg || !studioVideoPlayer) return;
  studioVideoPlayer.currentTime = seg.start;
  studioVideoPlayer.play();
  if (videoPlayPauseBtn) videoPlayPauseBtn.textContent = "⏸️ Pause";

  const onSegmentEnd = () => {
    if (studioVideoPlayer.currentTime >= seg.end) {
      studioVideoPlayer.pause();
      if (videoPlayPauseBtn) videoPlayPauseBtn.textContent = "⏯️ Play";
      studioVideoPlayer.removeEventListener("timeupdate", onSegmentEnd);
    }
  };
  studioVideoPlayer.addEventListener("timeupdate", onSegmentEnd);
};

window.updateSegmentText = (index, newText) => {
  if (parsedSegments[index]) {
    parsedSegments[index].text = newText;
    showSaveIndicator("Editing...");
  }
};

window.deleteSegment = (index) => {
  if (!confirm(`Delete subtitle line #${index + 1}?`)) return;
  parsedSegments.splice(index, 1);
  reindexSegments();
  renderSubtitleTable(parsedSegments);
  showSaveIndicator("Line deleted");
};

function reindexSegments() {
  parsedSegments.forEach((seg, i) => {
    seg.id = i + 1;
    seg.start_fmt = formatSecondsToSrt(seg.start);
    seg.end_fmt = formatSecondsToSrt(seg.end);
  });
}

// Add Cue button
if (addCueBtn) {
  addCueBtn.addEventListener("click", () => {
    const curTime = studioVideoPlayer ? studioVideoPlayer.currentTime : 0;
    const newCue = {
      id: parsedSegments.length + 1,
      start: roundMs(curTime),
      end: roundMs(curTime + 2.5),
      start_fmt: formatSecondsToSrt(curTime),
      end_fmt: formatSecondsToSrt(curTime + 2.5),
      text: "New subtitle text"
    };

    // Insert sorted by start time
    let inserted = false;
    for (let i = 0; i < parsedSegments.length; i++) {
      if (parsedSegments[i].start > newCue.start) {
        parsedSegments.splice(i, 0, newCue);
        inserted = true;
        break;
      }
    }
    if (!inserted) parsedSegments.push(newCue);

    reindexSegments();
    renderSubtitleTable(parsedSegments);
    showSaveIndicator("New line added");
  });
}

// Timing Sync Offset Shifter
if (shiftBackBtn) {
  shiftBackBtn.addEventListener("click", () => applySyncOffset(-0.5));
}
if (shiftFwdBtn) {
  shiftFwdBtn.addEventListener("click", () => applySyncOffset(0.5));
}
if (resetShiftBtn) {
  resetShiftBtn.addEventListener("click", () => {
    totalSyncOffset = 0.0;
    if (offsetDisplay) offsetDisplay.textContent = "0.0s";
    parsedSegments = JSON.parse(JSON.stringify(originalSegments));
    reindexSegments();
    renderSubtitleTable(parsedSegments);
    showSaveIndicator("Timing reset to original");
  });
}

function applySyncOffset(delta) {
  totalSyncOffset += delta;
  if (offsetDisplay) offsetDisplay.textContent = `${totalSyncOffset > 0 ? '+' : ''}${totalSyncOffset.toFixed(1)}s`;

  parsedSegments.forEach(seg => {
    seg.start = Math.max(0, roundMs(seg.start + delta));
    seg.end = Math.max(seg.start + 0.1, roundMs(seg.end + delta));
    seg.start_fmt = formatSecondsToSrt(seg.start);
    seg.end_fmt = formatSecondsToSrt(seg.end);
  });

  renderSubtitleTable(parsedSegments);
  showSaveIndicator(`Timing shifted ${totalSyncOffset > 0 ? '+' : ''}${totalSyncOffset.toFixed(1)}s`);
}

// Search filter
if (subtitleSearchInput) {
  subtitleSearchInput.addEventListener("input", (e) => {
    renderSubtitleTable(parsedSegments, e.target.value);
  });
}

// Copy All Text
if (copyAllBtn) {
  copyAllBtn.addEventListener("click", () => {
    const fullText = parsedSegments.map(s => `[${s.start_fmt}] ${s.text}`).join("\n");
    navigator.clipboard.writeText(fullText).then(() => {
      alert("All subtitle text copied to clipboard!");
    });
  });
}

// Save & Export Edited .SRT
if (saveEditsBtn) {
  saveEditsBtn.addEventListener("click", async () => {
    if (!currentJobId) return;

    saveEditsBtn.disabled = true;
    showSaveIndicator("Saving to server...");

    try {
      const res = await fetch(`/api/jobs/${currentJobId}/subtitles/update`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ segments: parsedSegments })
      });

      if (!res.ok) throw new Error("Failed to save updates");
      showSaveIndicator("✓ Saved & Updated!");

      // Trigger automatic download of updated .SRT file
      window.location.href = `/api/jobs/${currentJobId}/download/srt`;
    } catch (e) {
      alert("Save error: " + e.message);
      showSaveIndicator("Save failed");
    } finally {
      saveEditsBtn.disabled = false;
    }
  });
}

function showSaveIndicator(msg) {
  if (saveStatusBadge) {
    saveStatusBadge.textContent = msg;
    saveStatusBadge.style.color = "#3b82f6";
    setTimeout(() => {
      saveStatusBadge.textContent = "✓ Auto-Sync Active";
      saveStatusBadge.style.color = "#10b981";
    }, 2500);
  }
}

// Download Batch Zip
async function downloadBatchZip() {
  const completedJobIds = batchQueue
    .filter(item => item.status === "completed" && item.jobId)
    .map(item => item.jobId);

  if (completedJobIds.length === 0) {
    alert("No completed subtitle files to download.");
    return;
  }

  try {
    const res = await fetch("/api/batch/export-zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_ids: completedJobIds })
    });

    if (!res.ok) throw new Error("Batch zip generation failed");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "whisper_batch_subtitles.zip";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    alert("Batch download error: " + err.message);
  }
}

// =========================================================================
// 8. Timers & Terminal Logging Helpers
// =========================================================================
function startTimer() {
  stopTimer();
  timerStart = Date.now();
  if (elapsedTimer) elapsedTimer.textContent = "Elapsed: 00:00";
  timerInterval = setInterval(() => {
    const totalSecs = Math.floor((Date.now() - timerStart) / 1000);
    const m = String(Math.floor(totalSecs / 60)).padStart(2, "0");
    const s = String(totalSecs % 60).padStart(2, "0");
    if (elapsedTimer) elapsedTimer.textContent = `Elapsed: ${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  if (timerInterval) clearInterval(timerInterval);
  timerInterval = null;
}

function addLog(msg, type = "info") {
  if (!logTerminal) return;
  const line = document.createElement("div");
  line.className = `log-line ${type}`;

  const timeStr = new Date().toLocaleTimeString("en-US", { hour12: true, hour: "numeric", minute: "numeric", second: "numeric" });
  line.innerHTML = `<span class="log-ts">[${timeStr}]</span> ${escapeHtml(msg)}`;
  logTerminal.appendChild(line);
  logTerminal.scrollTop = logTerminal.scrollHeight;
}

function formatSecondsToSrt(seconds) {
  const totalMs = Math.max(0, Math.round(seconds * 1000));
  const ms = totalMs % 1000;
  const totalSecs = Math.floor(totalMs / 1000);
  const s = totalSecs % 60;
  const totalMins = Math.floor(totalSecs / 60);
  const m = totalMins % 60;
  const h = Math.floor(totalMins / 60);
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}

function roundMs(val) {
  return Math.round(val * 1000) / 1000;
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Cancel job button
if (cancelJobBtn) {
  cancelJobBtn.addEventListener("click", () => {
    if (confirm("Reset current view and start over?")) {
      stopTimer();
      if (progressSection) progressSection.classList.add("hidden");
      if (setupSection) setupSection.classList.remove("hidden");
      if (generateBtn) generateBtn.disabled = batchQueue.length === 0;
    }
  });
}

// =========================================================================
// 9. Startup
// =========================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupDropzone();
  initSystemStatus();
  setInterval(initSystemStatus, 8000);
});
