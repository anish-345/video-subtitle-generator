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
const selectedFileInfo = document.getElementById("selectedFileInfo");
const singleFileName = document.getElementById("singleFileName");
const singleFileSize = document.getElementById("singleFileSize");
const removeSingleFileBtn = document.getElementById("removeSingleFileBtn");

// Server Tasks & Subtitle History Elements
const tasksDashboardContainer = document.getElementById("tasksDashboardContainer");
const serverTasksHint = document.getElementById("serverTasksHint");
const serverTasksCount = document.getElementById("serverTasksCount");
const activeTasksBadge = document.getElementById("activeTasksBadge");
const refreshTasksBtn = document.getElementById("refreshTasksBtn");
const tasksList = document.getElementById("tasksList");

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

    // Rust Core Metrics & Queue Counter
    const effectiveQueued = (data.rust_core && data.rust_core.queued_jobs !== undefined)
      ? data.rust_core.queued_jobs
      : (data.queued_count || 0);
    if (queueBadge) queueBadge.textContent = `${effectiveQueued} in Queue`;

    if (data.rust_core && turboBadgeText) {
      turboBadgeText.textContent = data.rust_core.turbo_ready ? "170x Speed (10h in ~4m)" : "Queue Balanced";
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

  if (removeSingleFileBtn) {
    removeSingleFileBtn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      resetBatchQueue();
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
  if (selectedFileInfo) selectedFileInfo.classList.add("hidden");
  if (dropzonePrompt) dropzonePrompt.classList.remove("hidden");
  if (batchQueueContainer) batchQueueContainer.classList.add("hidden");
  if (generateBtn) {
    generateBtn.disabled = true;
    if (generateBtnText) generateBtnText.textContent = "Generate Subtitles (.SRT)";
  }
}

function renderBatchQueueUI() {
  if (batchQueue.length === 0) {
    if (selectedFileInfo) selectedFileInfo.classList.add("hidden");
    if (dropzonePrompt) dropzonePrompt.classList.remove("hidden");
    if (batchQueueContainer) batchQueueContainer.classList.add("hidden");
    if (generateBtn) {
      generateBtn.disabled = true;
      if (generateBtnText) generateBtnText.textContent = "Generate Subtitles (.SRT)";
    }
    return;
  }

  // Single file selected -> Show clean single file card in dropzone
  if (batchQueue.length === 1) {
    if (dropzonePrompt) dropzonePrompt.classList.add("hidden");
    if (batchQueueContainer) batchQueueContainer.classList.add("hidden");
    if (selectedFileInfo) {
      selectedFileInfo.classList.remove("hidden");
      if (singleFileName) singleFileName.textContent = batchQueue[0].file.name;
      if (singleFileSize) singleFileSize.textContent = formatBytes(batchQueue[0].file.size);
    }
    if (generateBtn) {
      generateBtn.disabled = isProcessingBatch;
      if (generateBtnText) {
        generateBtnText.textContent = isProcessingBatch ? "Processing..." : "Generate Subtitles (.SRT)";
      }
    }
    return;
  }

  // 2 or more files -> Show batch queue container
  if (selectedFileInfo) selectedFileInfo.classList.add("hidden");
  if (dropzonePrompt) dropzonePrompt.classList.add("hidden");
  if (batchQueueContainer) batchQueueContainer.classList.remove("hidden");

  let totalBytes = 0;
  let completedCount = 0;

  if (batchFileList) {
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
  }

  if (batchCountBadge) batchCountBadge.textContent = `${batchQueue.length} files`;
  if (batchSizeHint) batchSizeHint.textContent = `Total: ${formatBytes(totalBytes)}`;

  if (generateBtn) {
    generateBtn.disabled = isProcessingBatch;
    if (generateBtnText) {
      generateBtnText.textContent = isProcessingBatch
        ? `Processing (${activeBatchIndex + 1}/${batchQueue.length})...`
        : `⚡ Process Batch (${batchQueue.length} Videos)`;
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
// 3. Batch Queue Pipelined Upload & Server Task Assigning
// =========================================================================
let activeUploadXhr = null;
let activeFocusJobId = null;

if (generateBtn) {
  generateBtn.addEventListener("click", () => {
    if (batchQueue.length === 0 || isProcessingBatch) return;
    startBatchProcessingPipeline();
  });
}

async function startBatchProcessingPipeline() {
  isProcessingBatch = true;
  if (generateBtn) generateBtn.disabled = true;
  if (progressSection) progressSection.classList.remove("hidden");
  startTimer();

  addLog(`[Batch Controller] Initializing pipelined upload & task queue for ${batchQueue.length} files...`);

  // Start the sequential upload loop: uploads file 1 -> assigns job -> uploads file 2 -> assigns job...
  runPipelinedUploads();
}

async function runPipelinedUploads() {
  for (let i = 0; i < batchQueue.length; i++) {
    const item = batchQueue[i];
    if (item.jobId || item.status === "completed") continue;

    activeBatchIndex = i;
    item.status = "uploading";
    item.uploadProgress = 0;
    renderBatchQueueUI();

    if (batchProgressPill) {
      batchProgressPill.textContent = `Uploading ${i + 1} of ${batchQueue.length}`;
      batchProgressPill.classList.remove("hidden");
    }

    if (statusMessage) {
      statusMessage.textContent = `Uploading "${item.file.name}" to Server...`;
    }
    addLog(`[Uploader] Uploading "${item.file.name}" (${formatBytes(item.file.size)})...`);

    try {
      const data = await uploadFileWithXhr(item, (pct) => {
        item.uploadProgress = pct;
        updateItemCardUI(item);
        if (!activeFocusJobId || activeFocusJobId === item.jobId) {
          if (progressBar) progressBar.style.width = `${pct}%`;
          if (progressPercent) progressPercent.textContent = `${pct}%`;
          if (statusMessage) statusMessage.textContent = `Uploading "${item.file.name}" (${pct}%)...`;
        }
      });

      // Server assigned job_id and placed it in the queue!
      item.jobId = data.job_id;
      item.status = data.status || "queued";
      item.uploadProgress = 100;
      addLog(`[Task Assigned] Job ID ${item.jobId} created on server for "${item.file.name}" (${item.status}).`, "success");

      // Auto-focus the first active job
      if (!activeFocusJobId) {
        activeFocusJobId = item.jobId;
        currentJobId = item.jobId;
        currentFile = item.file;
      }

      renderBatchQueueUI();
      fetchServerJobs(); // Update server dashboard immediately!

      // Immediately launch live tracker for this job
      trackJobStream(item);

    } catch (err) {
      item.status = "failed";
      item.error = err.message;
      addLog(`[Upload Error] "${item.file.name}": ${err.message}`, "error");
      renderBatchQueueUI();
    }
  }

  if (batchProgressPill) {
    batchProgressPill.textContent = `All files uploaded • GPU processing`;
  }
}

function uploadFileWithXhr(item, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    activeUploadXhr = xhr;

    const isTurbo = turboToggle ? turboToggle.checked : true;
    const useVad = coverageSelect ? coverageSelect.value === "vad" : false;
    const noiseVal = noiseReductionSelect ? noiseReductionSelect.value : "studio";

    const formData = new FormData();
    formData.append("file", item.file);
    formData.append("model_size", modelSelect.value);
    formData.append("language", languageSelect.value);
    formData.append("task", taskSelect.value);
    formData.append("turbo_mode", isTurbo ? "true" : "false");
    formData.append("vad_filter", useVad ? "true" : "false");
    formData.append("noise_reduction", noiseVal);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        const pct = Math.min(100, Math.round((e.loaded / e.total) * 100));
        onProgress(pct);
      }
    };

    xhr.onload = () => {
      activeUploadXhr = null;
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const res = JSON.parse(xhr.responseText);
          resolve(res);
        } catch {
          reject(new Error("Invalid response from server"));
        }
      } else {
        let errMsg = `Upload failed (${xhr.status})`;
        try {
          const err = JSON.parse(xhr.responseText);
          errMsg = err.detail || err.message || errMsg;
        } catch {}
        reject(new Error(errMsg));
      }
    };

    xhr.onerror = () => {
      activeUploadXhr = null;
      reject(new Error("Network connection error during upload"));
    };

    xhr.open("POST", "/api/transcribe");
    xhr.send(formData);
  });
}

function trackJobStream(item) {
  if (!item.jobId) return;

  const sse = new EventSource(`/api/jobs/${item.jobId}/stream`);

  sse.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.status) item.status = payload.status;
      if (payload.progress !== undefined) item.progress = payload.progress;
      if (payload.message) item.message = payload.message;

      updateItemCardUI(item);

      // If this item is currently focused in the live monitor
      if (activeFocusJobId === item.jobId) {
        handleProgressUpdate(payload, item);
      }

      if (payload.step === "completed") {
        sse.close();
        item.status = "completed";
        item.progress = 100;
        updateItemCardUI(item);
        addLog(`[GPU Finished] Transcription completed for "${item.file.name}"!`, "success");

        fetchJobResultWithRetry(item.jobId)
          .then(result => {
            item.result = result;
            renderBatchQueueUI();
            fetchServerJobs();

            // If studio is empty, or this is focused job, load into studio
            if (!currentJobData || activeFocusJobId === item.jobId) {
              currentFile = item.file;
              currentJobId = item.jobId;
              finishJob(result);
      playCompletionChime();
      showToast(`Transcription finished for "${item.file ? item.file.name : 'video'}"!`, "success", 4000);
            }
            checkAllBatchFinished();
          })
          .catch(e => {
            console.warn("Result retrieve notice", e);
            checkAllBatchFinished();
          });

      } else if (payload.step === "error") {
        sse.close();
        item.status = "failed";
        item.error = payload.message || "Failed";
        updateItemCardUI(item);
        addLog(`[Job Error] "${item.file.name}": ${item.error}`, "error");
        renderBatchQueueUI();
        fetchServerJobs();
        checkAllBatchFinished();
      }
    } catch (e) {
      console.warn("SSE parse error", e);
    }
  };

  sse.onerror = async () => {
    sse.close();
    // Poll fallback for this job
    const result = await pollJobStatusUntilDone(item.jobId, item);
    if (result) {
      item.status = "completed";
      item.progress = 100;
      item.result = result;
      if (!currentJobData || activeFocusJobId === item.jobId) {
        currentFile = item.file;
        currentJobId = item.jobId;
        finishJob(result);
      playCompletionChime();
      showToast(`Transcription finished for "${item.file ? item.file.name : 'video'}"!`, "success", 4000);
      }
    } else {
      item.status = "failed";
    }
    updateItemCardUI(item);
    renderBatchQueueUI();
    fetchServerJobs();
    checkAllBatchFinished();
  };
}

function checkAllBatchFinished() {
  const isAnyRunning = batchQueue.some(i => i.status === "uploading" || i.status === "queued" || i.status === "processing");
  if (!isAnyRunning && batchQueue.length > 0) {
    isProcessingBatch = false;
    stopTimer();
    if (generateBtn) {
      generateBtn.disabled = false;
      if (generateBtnText) {
        generateBtnText.textContent = `⚡ Process Batch (${batchQueue.length} Videos)`;
      }
    }
    if (batchProgressPill) batchProgressPill.classList.add("hidden");
    const completedCount = batchQueue.filter(i => i.status === "completed").length;
    if (statusMessage) {
      statusMessage.textContent = `Batch finished! ${completedCount} of ${batchQueue.length} videos transcribed.`;
    }
    renderBatchQueueUI();
    fetchServerJobs();
  }
}

function updateItemCardUI(item) {
  const card = document.getElementById(`item-${item.id}`);
  if (!card) return;

  card.className = `batch-file-card ${item.status}`;
  const rightEl = card.querySelector(".batch-file-right");
  if (!rightEl) return;

  let statusBadge = "";
  if (item.status === "uploading") {
    statusBadge = `<span class="badge" style="background:#e0f2fe; color:#0369a1;"><span class="engine-dot pulse" style="background:#0284c7; width:7px; height:7px;"></span> ⬆️ Uploading (${item.uploadProgress || 0}%)</span>`;
  } else if (item.status === "queued") {
    statusBadge = `<span class="badge" style="background:#fef3c7; color:#92400e;">⏳ Queued on Server ${item.jobId ? '(' + item.jobId + ')' : ''}</span>`;
  } else if (item.status === "processing") {
    statusBadge = `<span class="badge badge-turbo"><span class="engine-dot pulse" style="background:#ec4899; width:7px; height:7px;"></span> ⚡ Transcribing (${item.progress || 0}%)</span>`;
  } else if (item.status === "completed") {
    statusBadge = `<span class="badge badge-emerald">✓ Completed</span>`;
  } else if (item.status === "failed") {
    statusBadge = `<span class="badge" style="background:#fee2e2; color:#b91c1c;">✕ Error</span>`;
  } else {
    statusBadge = `<span class="badge">⏳ Ready</span>`;
  }

  const idx = batchQueue.indexOf(item);
  let actionBtnHtml = `
    <button type="button" class="btn-icon-sm delete" title="Remove from queue" onclick="removeFileFromQueue(${idx})">✕</button>
  `;
  if (item.status === "completed" && item.result) {
    actionBtnHtml = `
      <button type="button" class="btn btn-secondary btn-sm" onclick="loadCompletedJobToStudio(${idx})">Open in Studio</button>
      <a class="btn btn-secondary btn-sm" href="/api/jobs/${item.jobId}/download/srt" download title="Download .SRT">⬇️ SRT</a>
      ${actionBtnHtml}
    `;
  } else if (item.status === "processing" || item.status === "queued") {
    actionBtnHtml = `
      <button type="button" class="btn btn-secondary btn-sm" onclick="focusActiveJob('${item.id}')" title="Monitor live progress">👁️ Monitor</button>
      ${actionBtnHtml}
    `;
  }

  rightEl.innerHTML = `${statusBadge} ${actionBtnHtml}`;
}

window.focusActiveJob = (itemId) => {
  const item = batchQueue.find(i => i.id === itemId);
  if (!item) return;
  activeFocusJobId = item.jobId;
  currentJobId = item.jobId;
  currentFile = item.file;
  if (statusMessage) statusMessage.textContent = `Monitoring "${item.file.name}" (${item.status})...`;
  if (progressBar) progressBar.style.width = `${item.progress || 0}%`;
  if (progressPercent) progressPercent.textContent = `${item.progress || 0}%`;
  if (progressSection) {
    progressSection.classList.remove("hidden");
    progressSection.scrollIntoView({ behavior: "smooth" });
  }
};

async function fetchJobResultWithRetry(jobId, maxAttempts = 20, delayMs = 350) {
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const res = await fetch(`/api/jobs/${jobId}/status`);
      if (res.ok) {
        const data = await res.json();
        if (data && data.status === "completed" && data.result && data.result.segments) {
          return data.result;
        }
      }
    } catch (e) {
      console.warn(`Retry fetching result for ${jobId} (attempt ${i + 1}):`, e);
    }
    await new Promise(r => setTimeout(r, delayMs));
  }
  throw new Error("Job completed but subtitle result could not be retrieved from server.");
}

async function pollJobStatusUntilDone(jobId, item, maxAttempts = 120, intervalMs = 2000) {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, intervalMs));
    try {
      const res = await fetch(`/api/jobs/${jobId}/status`);
      if (!res.ok) continue;
      const data = await res.json();

      if (data.status === "completed" && data.result && data.result.segments) {
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
// 3b. Server Tasks Dashboard & Multi-Device Subtitle Viewer
// =========================================================================
let isFetchingServerJobs = false;

async function fetchServerJobs() {
  if (isFetchingServerJobs) return;
  isFetchingServerJobs = true;

  try {
    const res = await fetch("/api/jobs");
    if (!res.ok) return;
    const data = await res.json();

    const jobs = data.jobs || [];
    const activeCount = data.active_count || 0;
    const totalCount = data.total || jobs.length;

    if (serverTasksCount) serverTasksCount.textContent = totalCount;
    if (activeTasksBadge) {
      activeTasksBadge.textContent = `${activeCount} Active`;
      if (activeCount > 0) {
        activeTasksBadge.className = "badge badge-turbo";
      } else {
        activeTasksBadge.className = "badge badge-queue";
      }
    }

    renderServerTasksList(jobs);
  } catch (err) {
    console.warn("Failed to fetch server jobs:", err);
  } finally {
    isFetchingServerJobs = false;
  }
}

function renderServerTasksList(jobs) {
  if (!tasksList) return;

  if (!jobs || jobs.length === 0) {
    tasksList.innerHTML = `
      <div class="tasks-empty-hint">
        No tasks queued or completed yet on this server. Drag and drop video(s) above to start!
      </div>
    `;
    return;
  }

  // Preserve scroll position if user is scrolling
  const prevScrollTop = tasksList.scrollTop;

  tasksList.innerHTML = "";
  jobs.forEach(job => {
    const card = document.createElement("div");
    card.className = `task-item-card ${job.status}`;
    card.id = `server-task-${job.job_id}`;

    let statusPill = "";
    if (job.status === "processing") {
      statusPill = `<span class="badge badge-turbo"><span class="engine-dot pulse" style="background:#ec4899;width:7px;height:7px;"></span> ${job.progress || 0}% Transcribing</span>`;
    } else if (job.status === "queued") {
      const qPos = job.queue_position ? ` #${job.queue_position}` : "";
      statusPill = `<span class="badge" style="background:#fef3c7; color:#92400e;">⏳ Queued${qPos}</span>`;
    } else if (job.status === "completed") {
      statusPill = `<span class="badge badge-emerald">✓ Completed</span>`;
    } else if (job.status === "failed") {
      statusPill = `<span class="badge" style="background:#fee2e2; color:#b91c1c;">✕ Failed</span>`;
    }

    let actionBtns = "";
    if (job.status === "completed" || job.has_subtitles) {
      actionBtns = `
        <button type="button" class="btn btn-primary btn-sm" onclick="loadJobById('${job.job_id}')" title="Open subtitle editor & video player in Studio">
          📺 Open in Studio
        </button>
        <a class="btn btn-secondary btn-sm" href="/api/jobs/${job.job_id}/download/srt" download title="Download .SRT">
          ⬇️ SRT
        </a>
        <a class="btn btn-secondary btn-sm" href="/api/jobs/${job.job_id}/download/vtt" download title="Download .VTT">
          ⬇️ VTT
        </a>
      `;
    } else if (job.status === "processing" || job.status === "queued") {
      actionBtns = `
        <button type="button" class="btn btn-secondary btn-sm" onclick="attachToLiveJob('${job.job_id}', '${escapeHtml(job.filename)}')">
          👁️ Monitor
        </button>
      `;
    }

    const durationText = job.duration ? `${Math.round(job.duration)}s` : "";
    const speedText = job.speedup_factor ? `⚡ ${job.speedup_factor}x` : "";
    const metaParts = [
      job.model_size ? `Model: ${job.model_size}` : "",
      job.language ? `Lang: ${job.language}` : "",
      durationText,
      speedText
    ].filter(Boolean).join(" • ");

    card.innerHTML = `
      <div class="task-item-left">
        <span class="task-item-icon">${job.status === "processing" ? "⚡" : "🎬"}</span>
        <div class="task-item-meta">
          <div class="task-item-title" title="${escapeHtml(job.filename)}">${escapeHtml(job.filename)}</div>
          <div class="task-item-subtitle">${metaParts || "Whisper Task " + job.job_id}</div>
        </div>
      </div>
      <div class="task-item-right">
        ${statusPill}
        ${actionBtns}
      </div>
    `;

    tasksList.appendChild(card);
  });

  tasksList.scrollTop = prevScrollTop;
}

async function loadJobById(jobId) {
  try {
    addLog(`[Studio] Loading subtitles for job ${jobId} from server...`);
    const res = await fetch(`/api/jobs/${jobId}/status`);
    if (!res.ok) throw new Error("Could not load job data from server");
    const data = await res.json();

    if (data.status === "completed" && data.result) {
      currentJobId = jobId;
      currentFile = null; // Streams video from /api/jobs/${jobId}/video
      finishJob(data.result);
      if (progressSection) progressSection.classList.add("hidden");
      if (resultsSection) {
        resultsSection.classList.remove("hidden");
        resultsSection.scrollIntoView({ behavior: "smooth" });
      }
    } else {
      alert(`Job is currently ${data.status}. Progress: ${data.progress || 0}%`);
    }
  } catch (err) {
    alert("Failed to load subtitles: " + err.message);
  }
}

function attachToLiveJob(jobId, filename) {
  currentJobId = jobId;
  currentFile = null;
  if (progressSection) progressSection.classList.remove("hidden");
  if (statusMessage) statusMessage.textContent = `Attached to live transcription: "${filename}"...`;
  startTimer();

  const dummyItem = { jobId: jobId, file: { name: filename, size: 0 }, progress: 0 };
  waitForJobCompletion(jobId, dummyItem)
    .then(result => {
      finishJob(result);
      playCompletionChime();
      showToast(`Transcription finished for "${item.file ? item.file.name : 'video'}"!`, "success", 4000);
    })
    .catch(err => {
      alert("Job monitoring stopped: " + err.message);
    });
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
      "deep_clean": "🧠 AI: Isolated Vocals + Music Removed",
      "music_removal": "🎵 AI: Background Music Removed",
      "denoise_ai": "🤖 AI: Deep Learning Speech Denoised",
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
            if (typeof autoScrollEnabled === "undefined" || autoScrollEnabled) { row.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
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

// Global exports for inline HTML onclick handlers
window.loadJobById = loadJobById;
window.attachToLiveJob = attachToLiveJob;
window.fetchServerJobs = fetchServerJobs;

// Refresh button for server tasks
if (refreshTasksBtn) {
  refreshTasksBtn.addEventListener("click", () => {
    fetchServerJobs();
  });
}

// =========================================================================
// 9. Startup
// =========================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupDropzone();
  initSystemStatus();
  fetchServerJobs();
  setInterval(initSystemStatus, 8000);
  setInterval(fetchServerJobs, 4000);
});


// =========================================================================
// 10. Theme, Sound, Toasts, Stepper & Mobile Controller
// =========================================================================

// Theme State
let currentTheme = "light";
let soundEnabled = true;
let autoScrollEnabled = true;

// DOM Elements - Theme & Mobile
const themeToggleBtn = document.getElementById("themeToggleBtn");
const themeToggleIcon = document.getElementById("themeToggleIcon");
const soundToggleBtn = document.getElementById("soundToggleBtn");
const soundToggleIcon = document.getElementById("soundToggleIcon");

const stepBtnSetup = document.getElementById("stepBtnSetup");
const stepBtnProgress = document.getElementById("stepBtnProgress");
const stepBtnStudio = document.getElementById("stepBtnStudio");
const stepBatchCount = document.getElementById("stepBatchCount");
const stepProgressBadge = document.getElementById("stepProgressBadge");
const stepStudioBadge = document.getElementById("stepStudioBadge");

const mobileBottomBar = document.getElementById("mobileBottomBar");
const mobileGenerateBtn = document.getElementById("mobileGenerateBtn");
const mobileGenerateBtnText = document.getElementById("mobileGenerateBtnText");
const mobileStudioActions = document.getElementById("mobileStudioActions");
const mobileDownloadSrtBtn = document.getElementById("mobileDownloadSrtBtn");
const mobileSaveEditsBtn = document.getElementById("mobileSaveEditsBtn");

const toggleAutoScrollBtn = document.getElementById("toggleAutoScrollBtn");
const overlayStyleSelect = document.getElementById("overlayStyleSelect");
const overlaySizeSelect = document.getElementById("overlaySizeSelect");

// --- Theme Management ---
function initTheme() {
  const saved = localStorage.getItem("whisper_theme");
  if (saved) {
    currentTheme = saved;
  } else if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
    currentTheme = "dark";
  } else {
    currentTheme = "light";
  }
  applyTheme(currentTheme);
}

function applyTheme(theme) {
  currentTheme = theme;
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("whisper_theme", theme);
  if (themeToggleIcon) {
    themeToggleIcon.textContent = theme === "dark" ? "☀️" : "🌙";
  }
  if (themeToggleBtn) {
    themeToggleBtn.title = theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode";
  }
}

if (themeToggleBtn) {
  themeToggleBtn.addEventListener("click", () => {
    applyTheme(currentTheme === "dark" ? "light" : "dark");
    showToast(`Switched to ${currentTheme === "dark" ? "Dark" : "Light"} Mode`, "info", 2000);
  });
}

// --- Sound Chime (Web Audio API) ---
function initSound() {
  const saved = localStorage.getItem("whisper_sound");
  soundEnabled = saved !== "off";
  if (soundToggleIcon) {
    soundToggleIcon.textContent = soundEnabled ? "🔔" : "🔕";
  }
}

if (soundToggleBtn) {
  soundToggleBtn.addEventListener("click", () => {
    soundEnabled = !soundEnabled;
    localStorage.setItem("whisper_sound", soundEnabled ? "on" : "off");
    if (soundToggleIcon) soundToggleIcon.textContent = soundEnabled ? "🔔" : "🔕";
    showToast(soundEnabled ? "Audio chime enabled 🔔" : "Audio chime muted 🔕", "info", 2000);
    if (soundEnabled) playCompletionChime();
  });
}

function playCompletionChime() {
  if (!soundEnabled) return;
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    const now = ctx.currentTime;

    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = "sine";
    osc1.frequency.setValueAtTime(659.25, now); // E5
    gain1.gain.setValueAtTime(0.12, now);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(now);
    osc1.stop(now + 0.35);

    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = "sine";
    osc2.frequency.setValueAtTime(987.77, now + 0.15); // B5
    gain2.gain.setValueAtTime(0.15, now + 0.15);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.65);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now + 0.15);
    osc2.stop(now + 0.65);
  } catch (e) {
    console.warn("Audio chime notice", e);
  }
}

// --- Toast Notification System ---
function showToast(message, type = "info", duration = 3500) {
  const container = document.getElementById("toastContainer");
  if (!container) return;
  const toast = document.createElement("div");
  toast.className = `toast-item toast-${type}`;

  const icons = {
    success: "✓",
    error: "✕",
    warning: "⚠️",
    info: "ℹ️"
  };

  toast.innerHTML = `
    <span class="toast-icon">${icons[type] || "ℹ️"}</span>
    <span class="toast-msg">${escapeHtml(message)}</span>
    <button type="button" class="toast-close" aria-label="Close">✕</button>
  `;

  toast.querySelector(".toast-close").onclick = () => {
    toast.classList.add("toast-exit");
    setTimeout(() => toast.remove(), 250);
  };

  container.appendChild(toast);
  setTimeout(() => toast.classList.add("toast-visible"), 15);

  setTimeout(() => {
    if (toast.parentNode) {
      toast.classList.add("toast-exit");
      setTimeout(() => toast.remove(), 250);
    }
  }, duration);
}

// --- Studio Stepper Navigation ---
function initStepper() {
  const steps = [
    { btn: stepBtnSetup, section: setupSection },
    { btn: stepBtnProgress, section: progressSection },
    { btn: stepBtnStudio, section: resultsSection }
  ];

  steps.forEach(s => {
    if (!s.btn) return;
    s.btn.addEventListener("click", () => {
      // Allow switching between active views
      if (s.section === progressSection && !isProcessingBatch && (!currentJobId || currentJobData)) {
        showToast("No active GPU transcription currently in progress.", "info");
      }
      if (s.section === resultsSection && !currentJobData) {
        showToast("No completed subtitles yet. Upload a video to get started!", "info");
        return;
      }
      activateSection(s.section);
    });
  });
}

function activateSection(targetSection) {
  if (setupSection) setupSection.classList.toggle("hidden", setupSection !== targetSection);
  if (progressSection) progressSection.classList.toggle("hidden", progressSection !== targetSection);
  if (resultsSection) resultsSection.classList.toggle("hidden", resultsSection !== targetSection);

  if (stepBtnSetup) stepBtnSetup.classList.toggle("active", setupSection === targetSection);
  if (stepBtnProgress) stepBtnProgress.classList.toggle("active", progressSection === targetSection);
  if (stepBtnStudio) stepBtnStudio.classList.toggle("active", resultsSection === targetSection);

  // Sync mobile bottom bar
  updateMobileBottomBar();
}

function updateStepperBadges() {
  if (stepBatchCount) {
    stepBatchCount.textContent = batchQueue.length.toString();
  }
  if (stepStudioBadge && currentJobData) {
    stepStudioBadge.classList.remove("hidden");
  }
}

// --- Quick Presets ---
function initPresets() {
  const presetPills = document.querySelectorAll(".preset-pill");
  presetPills.forEach(pill => {
    pill.addEventListener("click", () => {
      presetPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const preset = pill.dataset.preset;
      applyPreset(preset);
    });
  });
}

function applyPreset(preset) {
  if (preset === "turbo") {
    if (modelSelect) modelSelect.value = "base";
    if (noiseReductionSelect) noiseReductionSelect.value = "studio";
    if (turboToggle) turboToggle.checked = true;
    if (coverageSelect) coverageSelect.value = "full";
    showToast("⚡ Preset applied: 170x Turbo (Fast)", "success");
  } else if (preset === "deep_clean") {
    if (modelSelect) modelSelect.value = "base";
    if (noiseReductionSelect) noiseReductionSelect.value = "deep_clean";
    if (turboToggle) turboToggle.checked = true;
    if (coverageSelect) coverageSelect.value = "full";
    showToast("🧠 Preset applied: AI Voice Isolation & Music Removal (UVR MDX-Net + GTCRN)", "success");
  } else if (preset === "sensevoice") {
    if (modelSelect) modelSelect.value = "sensevoice-small";
    if (noiseReductionSelect) noiseReductionSelect.value = "deep_clean";
    if (turboToggle) turboToggle.checked = true;
    if (coverageSelect) coverageSelect.value = "full";
    showToast("🎯 Preset applied: Alibaba SenseVoice-Small with Deep Clean", "success");
  } else if (preset === "cinema") {
    if (modelSelect) modelSelect.value = "large-v3-turbo";
    if (noiseReductionSelect) noiseReductionSelect.value = "deep_clean";
    if (turboToggle) turboToggle.checked = true;
    if (coverageSelect) coverageSelect.value = "full";
    showToast("👑 Preset applied: Cinema SOTA (Large-v3 Turbo + Deep Clean)", "success");
  }
}

// --- Mobile Bottom Bar Sync ---
function updateMobileBottomBar() {
  if (!mobileBottomBar) return;

  const isStudioVisible = resultsSection && !resultsSection.classList.contains("hidden");

  if (isStudioVisible) {
    if (mobileGenerateBtn) mobileGenerateBtn.classList.add("hidden");
    if (mobileStudioActions) mobileStudioActions.classList.remove("hidden");
  } else {
    if (mobileGenerateBtn) mobileGenerateBtn.classList.remove("hidden");
    if (mobileStudioActions) mobileStudioActions.classList.add("hidden");
    if (mobileGenerateBtn && generateBtn) {
      mobileGenerateBtn.disabled = generateBtn.disabled;
      if (mobileGenerateBtnText && generateBtnText) {
        mobileGenerateBtnText.textContent = generateBtnText.textContent;
      }
    }
  }
}

if (mobileGenerateBtn) {
  mobileGenerateBtn.addEventListener("click", () => {
    if (generateBtn && !generateBtn.disabled) {
      generateBtn.click();
    }
  });
}

if (mobileDownloadSrtBtn) {
  mobileDownloadSrtBtn.addEventListener("click", () => {
    if (downloadSrtBtn) downloadSrtBtn.click();
  });
}

if (mobileSaveEditsBtn) {
  mobileSaveEditsBtn.addEventListener("click", () => {
    if (saveEditsBtn) saveEditsBtn.click();
  });
}

// --- Auto-Scroll Toggle ---
if (toggleAutoScrollBtn) {
  toggleAutoScrollBtn.addEventListener("click", () => {
    autoScrollEnabled = !autoScrollEnabled;
    toggleAutoScrollBtn.textContent = autoScrollEnabled ? "📜 Auto-Scroll: ON" : "📜 Auto-Scroll: OFF";
    toggleAutoScrollBtn.classList.toggle("active", autoScrollEnabled);
    showToast(autoScrollEnabled ? "Auto-scroll enabled" : "Auto-scroll paused", "info", 1800);
  });
}

// --- Overlay Style & Size Customizer ---
if (overlayStyleSelect) {
  overlayStyleSelect.addEventListener("change", (e) => {
    if (!videoSubtitleOverlay) return;
    videoSubtitleOverlay.classList.remove("theme-netflix", "theme-yellow", "theme-white");
    videoSubtitleOverlay.classList.add(`theme-${e.target.value}`);
  });
}

if (overlaySizeSelect) {
  overlaySizeSelect.addEventListener("change", (e) => {
    if (!videoSubtitleOverlay) return;
    videoSubtitleOverlay.classList.remove("size-small", "size-medium", "size-large");
    videoSubtitleOverlay.classList.add(`size-${e.target.value}`);
  });
}

// Hook into startup initialization
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initSound();
  initStepper();
  initPresets();
  updateStepperBadges();
  updateMobileBottomBar();
});
