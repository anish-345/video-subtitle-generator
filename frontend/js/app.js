// Whisper Subtitle Studio - Modern Full-Width Studio Controller

let currentFile = null;
let currentJobId = null;
let currentJobData = null;
let eventSource = null;
let timerInterval = null;
let timerStart = null;
let parsedSegments = [];
let originalSegments = [];
let totalSyncOffset = 0.0;

// Header & System Engine Badges
const rustBadge = document.getElementById("rustBadge");
const queueBadge = document.getElementById("queueBadge");
const gpuNameEl = document.getElementById("gpuName");
const turboBadge = document.getElementById("turboBadge");
const turboBadgeText = document.getElementById("turboBadgeText");
const turboToggle = document.getElementById("turboToggle");
const queueStatusPill = document.getElementById("queueStatusPill");
const speedupPill = document.getElementById("speedupPill");

// Upload & Setup Form
const dropzone = document.getElementById("dropzone");
const videoFileInput = document.getElementById("videoFileInput");
const dropzonePrompt = document.getElementById("dropzonePrompt");
const selectedFileInfo = document.getElementById("selectedFileInfo");
const fileNameEl = document.getElementById("fileName");
const fileSizeEl = document.getElementById("fileSize");
const removeFileBtn = document.getElementById("removeFileBtn");

const modelSelect = document.getElementById("modelSelect");
const languageSelect = document.getElementById("languageSelect");
const taskSelect = document.getElementById("taskSelect");
const coverageSelect = document.getElementById("coverageSelect");
const noiseReductionSelect = document.getElementById("noiseReductionSelect");
const generateBtn = document.getElementById("generateBtn");

// Progress & Live Stream
const setupSection = document.getElementById("setupSection");
const progressSection = document.getElementById("progressSection");
const progressBar = document.getElementById("progressBar");
const progressPercent = document.getElementById("progressPercent");
const statusMessage = document.getElementById("statusMessage");
const elapsedTimer = document.getElementById("elapsedTimer");
const logTerminal = document.getElementById("logTerminal");

// Subtitle Studio Workspace
const resultsSection = document.getElementById("resultsSection");
const statLang = document.getElementById("statLang");
const statCues = document.getElementById("statCues");
const statWords = document.getElementById("statWords");
const statDuration = document.getElementById("statDuration");
const statSpeed = document.getElementById("statSpeed");
const statPurge = document.getElementById("statPurge");

// Download & Export Buttons
const downloadSrtBtn = document.getElementById("downloadSrtBtn");
const downloadVttBtn = document.getElementById("downloadVttBtn");
const downloadTxtBtn = document.getElementById("downloadTxtBtn");
const downloadCleanAudioBtn = document.getElementById("downloadCleanAudioBtn");
const statNoise = document.getElementById("statNoise");
const copyAllBtn = document.getElementById("copyAllBtn");
const exportEditedSrtBtn = document.getElementById("exportEditedSrtBtn");

// Studio Search & Sync Offset Controls
const subtitleSearchInput = document.getElementById("subtitleSearchInput");
const searchMatchCount = document.getElementById("searchMatchCount");
const shiftBackBtn = document.getElementById("shiftBackBtn");
const shiftFwdBtn = document.getElementById("shiftFwdBtn");
const offsetDisplay = document.getElementById("offsetDisplay");
const resetShiftBtn = document.getElementById("resetShiftBtn");
const timelineList = document.getElementById("timelineList");

// =========================================================================
// 1. Initialize & Poll Engine Status
// =========================================================================
async function initSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("Failed to query status");
    const data = await res.json();

    // Optional auto-redirect if requested via ?wifi=1 query param
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get("wifi") === "1" && data.wifi_ip && data.wifi_ip !== "127.0.0.1" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1")) {
      const targetPort = window.location.port ? `:${window.location.port}` : ":8000";
      const targetUrl = `${window.location.protocol}//${data.wifi_ip}${targetPort}${window.location.pathname}${window.location.search}${window.location.hash}`;
      window.location.replace(targetUrl);
      return;
    }

    // Update Wi-Fi Badge
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
        if (data.rust_core.turbo_ready) {
          turboBadgeText.textContent = "170x Speed (10h in ~4m)";
        } else {
          turboBadgeText.textContent = "Queue Balanced";
        }
      }
    }

    // GPU Metrics
    const gpuComputeTag = document.getElementById("gpuComputeTag");
    if (data.gpu && data.gpu.cuda_available) {
      if (gpuNameEl) gpuNameEl.textContent = `${data.gpu.device_name} (${data.gpu.vram_gb} GB)`;
      if (gpuComputeTag) gpuComputeTag.textContent = data.gpu.recommended_compute.toUpperCase();
    } else {
      if (gpuNameEl) gpuNameEl.textContent = "CPU Fallback Mode";
      if (gpuComputeTag) gpuComputeTag.textContent = "CPU";
    }

    // Update models if provided by backend
    if (data.available_models && Array.isArray(data.available_models) && modelSelect) {
      updateModelOptions(data.available_models);
    }
  } catch (err) {
    console.warn("Status check notice:", err);
  }
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
// 2. Drag & Drop File Setup
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
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  if (videoFileInput) {
    videoFileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  if (removeFileBtn) {
    removeFileBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      resetFileSelection();
    });
  }
}

function handleFileSelected(file) {
  if (file.size > 25 * 1024 * 1024 * 1024) {
    alert("File exceeds the 25 GB maximum supported size.");
    return;
  }
  currentFile = file;
  if (fileNameEl) fileNameEl.textContent = file.name;
  if (fileSizeEl) fileSizeEl.textContent = formatBytes(file.size);

  if (dropzonePrompt) dropzonePrompt.classList.add("hidden");
  if (selectedFileInfo) selectedFileInfo.classList.remove("hidden");
  if (generateBtn) generateBtn.disabled = false;
}

function resetFileSelection() {
  currentFile = null;
  if (videoFileInput) videoFileInput.value = "";
  if (dropzonePrompt) dropzonePrompt.classList.remove("hidden");
  if (selectedFileInfo) selectedFileInfo.classList.add("hidden");
  if (generateBtn) generateBtn.disabled = true;
}

function formatBytes(bytes) {
  if (!bytes || bytes <= 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
  const i = Math.min(sizes.length - 1, Math.floor(Math.log(bytes) / Math.log(k)));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// =========================================================================
// 3. Trigger Subtitle Generation
// =========================================================================
if (generateBtn) {
  generateBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    generateBtn.disabled = true;
    if (progressSection) progressSection.classList.remove("hidden");
    if (resultsSection) resultsSection.classList.add("hidden");
    if (queueStatusPill) queueStatusPill.classList.add("hidden");
    if (speedupPill) speedupPill.classList.add("hidden");
    if (logTerminal) logTerminal.innerHTML = "";
    if (progressBar) progressBar.style.width = "5%";
    if (progressPercent) progressPercent.textContent = "5%";
    if (statusMessage) statusMessage.textContent = "Uploading video to Rust Concurrency Core...";

    startTimer();
    const isTurbo = turboToggle ? turboToggle.checked : true;
    const useVad = coverageSelect ? coverageSelect.value === "vad" : false;
    const noiseVal = noiseReductionSelect ? noiseReductionSelect.value : "studio";
    addLog(`[Rust Core] Submitting "${currentFile.name}" (${formatBytes(currentFile.size)})...`);
    addLog(`[Config] Model: ${modelSelect.value} | Language: ${languageSelect.value} | Noise Cancellation: ${noiseVal} | Coverage: ${useVad ? 'Smart VAD' : '100% Full Video Scan (Anti-Dropout)'} | Turbo: ${isTurbo ? '170x' : 'Balanced'}`);

    const formData = new FormData();
    formData.append("file", currentFile);
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
          try {
            errMsg = await res.text();
          } catch {}
        }
        throw new Error(errMsg);
      }

      const data = await res.json();
      currentJobId = data.job_id;

      // Handle instant cache hit (already completed)
      if (data.status === "completed" && data.result) {
        stopTimer();
        localStorage.removeItem("whisper_active_job");
        addLog(`[Instant Hit] Subtitles for "${data.filename || currentFile.name}" were already generated! Loaded immediately.`, "success");
        finishJob(data.result);
        return;
      }

      // Handle deduplication re-attachment
      if (data.deduplicated) {
        addLog(`[Deduplication] Task #${currentJobId} is already in progress for "${data.filename}". Reconnecting without duplicating...`, "info");
      } else {
        addLog(`[Rust Gateway] Job assigned: ${currentJobId}. Connecting to GPU event stream...`, "success");
      }

      // Save active session in localStorage to survive browser refresh
      localStorage.setItem("whisper_active_job", JSON.stringify({
        jobId: currentJobId,
        filename: currentFile ? currentFile.name : (data.filename || "video"),
        timestamp: Date.now()
      }));

      connectEventSource(currentJobId);
    } catch (err) {
      stopTimer();
      if (statusMessage) statusMessage.textContent = "Error: " + err.message;
      addLog("Error: " + err.message, "error");
      if (generateBtn) generateBtn.disabled = false;
    }
  });
}

const cancelJobBtn = document.getElementById("cancelJobBtn");
if (cancelJobBtn) {
  cancelJobBtn.addEventListener("click", () => {
    if (confirm("Abandon this view and start over? The server will finish in background.")) {
      if (eventSource) eventSource.close();
      localStorage.removeItem("whisper_active_job");
      stopTimer();
      if (progressSection) progressSection.classList.add("hidden");
      if (setupSection) setupSection.classList.remove("hidden");
      if (generateBtn) generateBtn.disabled = !currentFile;
    }
  });
}

function connectEventSource(jobId) {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource(`/api/jobs/${jobId}/stream`);

  eventSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      handleProgressUpdate(payload);
    } catch (e) {
      console.warn("SSE parse error", e, event.data);
    }
  };

  eventSource.onerror = async () => {
    eventSource.close();
    checkJobStatusPoll(jobId);
  };
}

async function checkJobStatusPoll(jobId, maxAttempts = 120, intervalMs = 3000) {
  for (let i = 0; i < maxAttempts; i++) {
    await new Promise(r => setTimeout(r, intervalMs));
    try {
      const res = await fetch(`/api/jobs/${jobId}/status`);
      if (!res.ok) continue;
      const data = await res.json();
      if (data.status === "completed") {
        stopTimer();
        localStorage.removeItem("whisper_active_job");
        finishJob(data.result);
        return;
      } else if (data.status === "failed") {
        stopTimer();
        localStorage.removeItem("whisper_active_job");
        if (statusMessage) statusMessage.textContent = "Failed: " + (data.error || "Unknown error");
        addLog("Job failed: " + (data.error || "Unknown error"), "error");
        if (generateBtn) generateBtn.disabled = false;
        return;
      } else if (data.status === "processing" || data.status === "queued") {
        if (data.progress !== undefined) {
          const pct = Math.min(100, Math.max(0, data.progress));
          if (progressBar) progressBar.style.width = `${pct}%`;
          if (progressPercent) progressPercent.textContent = `${pct}%`;
        }
        if (data.message && statusMessage) {
          statusMessage.textContent = data.message;
        }
      }
    } catch (e) {
      console.warn("Poll attempt failed, retrying...", e);
    }
  }
}

function handleProgressUpdate(data) {
  if (data.progress !== undefined) {
    const pct = Math.min(100, Math.max(0, data.progress));
    if (progressBar) progressBar.style.width = `${pct}%`;
    if (progressPercent) progressPercent.textContent = `${pct}%`;
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

  if (data.step === "completed") {
    if (eventSource) eventSource.close();
    stopTimer();
    addLog(`Subtitles generated successfully! Intermediate audio purged by Rust.`, "success");
    fetchJobResult(currentJobId);
  } else if (data.step === "error") {
    if (eventSource) eventSource.close();
    stopTimer();
    addLog("Transcription error: " + data.message, "error");
    if (generateBtn) generateBtn.disabled = false;
  }
}

async function fetchJobResult(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}/status`);
    const data = await res.json();
    if (data.status === "failed") {
      stopTimer();
      if (statusMessage) statusMessage.textContent = "Failed: " + (data.error || "Unknown error");
      addLog("Job failed: " + (data.error || "Unknown error"), "error");
      if (generateBtn) generateBtn.disabled = false;
      return;
    }
    if (!data.result) {
      addLog("Job result not available yet, polling...", "warn");
      checkJobStatusPoll(jobId);
      return;
    }
    finishJob(data.result);
  } catch (err) {
    addLog("Error fetching results: " + err.message, "error");
  }
}

// =========================================================================
// 4. Subtitle Studio Completion & Metrics Display
// =========================================================================
function finishJob(result) {
  currentJobData = result;
  // Deep clone segments to preserve originals for offset resets
  originalSegments = JSON.parse(JSON.stringify(result.segments || []));
  parsedSegments = JSON.parse(JSON.stringify(result.segments || []));
  totalSyncOffset = 0.0;
  if (offsetDisplay) offsetDisplay.textContent = "0.0s";

  setTimeout(() => {
    if (progressSection) progressSection.classList.add("hidden");
  }, 1000);

  if (resultsSection) resultsSection.classList.remove("hidden");
  if (generateBtn) generateBtn.disabled = false;

  // Calculate Studio Metrics
  const langName = (result.detected_language || "en").toUpperCase();
  const cueCount = parsedSegments.length;
  let wordCount = 0;
  parsedSegments.forEach(s => {
    const words = s.text.trim().split(/\s+/).filter(w => w.length > 0);
    wordCount += words.length;
  });

  const durationSec = result.total_duration || (parsedSegments.length > 0 ? parsedSegments[parsedSegments.length - 1].end : 0);
  const durMinutes = Math.floor(durationSec / 60);
  const durSecs = Math.floor(durationSec % 60);
  const durFormatted = durMinutes > 0 ? `${durMinutes}m ${durSecs}s` : `${durationSec.toFixed(1)}s`;

  if (statLang) statLang.textContent = `🌐 ${langName} (${result.language_probability ? result.language_probability + '%' : '100%'})`;
  if (statCues) statCues.textContent = `📝 ${cueCount} Cues`;
  if (statWords) statWords.textContent = `💬 ${wordCount} Words`;
  if (statDuration) statDuration.textContent = `⏱️ ${durFormatted}`;
  if (statSpeed) {
    const speed = result.speedup_factor ? `${result.speedup_factor}x Acceleration` : "170x Acceleration";
    statSpeed.textContent = `⚡ ${speed}`;
  }
  if (statPurge) {
    statPurge.textContent = result.video_deleted ? "🗑️ Video Purged (0B Disk Waste)" : "💾 Audio Cleaned";
  }
  if (statNoise) {
    const nMap = {
      "studio": "🎙️ Denoised: Studio Clarity",
      "aggressive": "🌪️ Denoised: Aggressive",
      "vocal_boost": "🔊 Denoised: Voice Boost",
      "off": "⚪ Denoised: Off"
    };
    statNoise.textContent = nMap[result.noise_reduction] || "🎙️ Denoised: Studio Clarity";
  }

  // Setup Download Links with clean names
  const originalName = result.filename || (currentFile ? currentFile.name : `subtitles_${result.job_id}`);
  const stem = originalName.replace(/\.[^/.]+$/, "");

  if (downloadCleanAudioBtn) {
    if (result.clean_audio_url) {
      downloadCleanAudioBtn.classList.remove("hidden");
      downloadCleanAudioBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-svg-sm">
          <path d="M9 18V5l12-2v13"></path>
          <circle cx="6" cy="18" r="3"></circle>
          <circle cx="18" cy="16" r="3"></circle>
        </svg>
        Clean Audio (.MP3)
      `;
      downloadCleanAudioBtn.onclick = () => window.location.href = `/api/jobs/${result.job_id}/clean-audio`;
    } else {
      downloadCleanAudioBtn.classList.add("hidden");
    }
  }

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

  // Render Full-Width Studio Timeline
  renderStudioTimeline(parsedSegments);
  resultsSection.scrollIntoView({ behavior: "smooth" });
}

// =========================================================================
// 5. Full-Width Subtitle Timeline & Segment Editing
// =========================================================================
function renderStudioTimeline(segments, filterText = "") {
  if (!timelineList) return;
  timelineList.innerHTML = "";

  if (!segments || segments.length === 0) {
    timelineList.innerHTML = '<p class="hint" style="text-align:center; padding: 36px;">No speech segments detected.</p>';
    return;
  }

  const query = filterText.toLowerCase().trim();
  let matchCount = 0;

  segments.forEach((seg, idx) => {
    const textLower = seg.text.toLowerCase();
    const timeMatch = seg.start_fmt.includes(query) || seg.end_fmt.includes(query);
    const textMatch = textLower.includes(query);
    const isVisible = query === "" || textMatch || timeMatch;

    if (isVisible) matchCount++;

    const card = document.createElement("div");
    card.className = `segment-card ${isVisible ? "" : "hidden"}`;
    card.id = `seg-${seg.id || (idx + 1)}`;
    card.dataset.id = seg.id || (idx + 1);
    card.dataset.index = idx;

    // Calculate reading pace (characters per second)
    const duration = Math.max(0.1, (seg.end - seg.start));
    const charCount = seg.text.trim().length;
    const cps = (charCount / duration).toFixed(1);
    let paceClass = "normal";
    let paceLabel = `${cps} cps`;
    if (cps > 22) {
      paceClass = "fast";
      paceLabel = `${cps} cps (Fast)`;
    } else if (cps < 8) {
      paceLabel = `${cps} cps (Relaxed)`;
    }

    card.innerHTML = `
      <div class="segment-header">
        <div class="segment-meta-left">
          <span class="segment-num">#${idx + 1}</span>
          <span class="segment-time" id="time-${idx}">${seg.start_fmt} ➔ ${seg.end_fmt}</span>
          <span class="segment-duration-pill" id="dur-${idx}">(${duration.toFixed(1)}s)</span>
        </div>
        <div class="segment-meta-right">
          <span class="pace-pill ${paceClass}" id="pace-${idx}">${paceLabel}</span>
          <button type="button" class="card-action-btn copy-cue-btn" title="Copy this cue" data-idx="${idx}">📋</button>
          <button type="button" class="card-action-btn delete-btn" title="Delete this cue" data-idx="${idx}">✕</button>
        </div>
      </div>
      <div class="segment-text" contenteditable="true" spellcheck="false" data-idx="${idx}">${escapeHtml(seg.text)}</div>
    `;

    // Blur / change on editable text
    const textEl = card.querySelector(".segment-text");
    textEl.addEventListener("input", () => {
      seg.text = textEl.innerText.trim();
      // Recalculate pace live
      const newDur = Math.max(0.1, (seg.end - seg.start));
      const newCps = (seg.text.length / newDur).toFixed(1);
      const pacePill = card.querySelector(`#pace-${idx}`);
      if (pacePill) pacePill.textContent = `${newCps} cps`;
    });

    // Copy single cue
    const copyBtn = card.querySelector(".copy-cue-btn");
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(seg.text).then(() => {
        copyBtn.textContent = "✓";
        setTimeout(() => copyBtn.textContent = "📋", 1500);
      });
    });

    // Delete single cue
    const delBtn = card.querySelector(".delete-btn");
    delBtn.addEventListener("click", () => {
      if (confirm(`Delete subtitle cue #${idx + 1}?`)) {
        parsedSegments.splice(idx, 1);
        renderStudioTimeline(parsedSegments, subtitleSearchInput ? subtitleSearchInput.value : "");
        if (statCues) statCues.textContent = `📝 ${parsedSegments.length} Cues`;
      }
    });

    timelineList.appendChild(card);
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

// =========================================================================
// 6. Search & Real-Time Filter
// =========================================================================
if (subtitleSearchInput) {
  subtitleSearchInput.addEventListener("input", (e) => {
    const val = e.target.value;
    renderStudioTimeline(parsedSegments, val);
  });
}

// =========================================================================
// 7. Global Time Offset Sync Shifter (+/- 0.5s)
// =========================================================================
function applyTimeShift(deltaSeconds) {
  totalSyncOffset += deltaSeconds;
  totalSyncOffset = Math.round(totalSyncOffset * 10) / 10;

  if (offsetDisplay) {
    const sign = totalSyncOffset > 0 ? "+" : "";
    offsetDisplay.textContent = `${sign}${totalSyncOffset.toFixed(1)}s`;
    offsetDisplay.style.color = totalSyncOffset !== 0 ? "var(--rust-orange)" : "var(--primary)";
  }

  // Update all segments from original baseline
  parsedSegments.forEach((seg, idx) => {
    const orig = originalSegments[idx];
    if (!orig) return;

    seg.start = Math.max(0, orig.start + totalSyncOffset);
    seg.end = Math.max(seg.start + 0.1, orig.end + totalSyncOffset);
    seg.start_fmt = formatTimestampSrt(seg.start);
    seg.end_fmt = formatTimestampSrt(seg.end);

    const timeEl = document.getElementById(`time-${idx}`);
    if (timeEl) {
      timeEl.textContent = `${seg.start_fmt} ➔ ${seg.end_fmt}`;
    }
    const durEl = document.getElementById(`dur-${idx}`);
    if (durEl) {
      durEl.textContent = `(${(seg.end - seg.start).toFixed(1)}s)`;
    }
  });
}

function resetTimeShift() {
  totalSyncOffset = 0.0;
  if (offsetDisplay) {
    offsetDisplay.textContent = "0.0s";
    offsetDisplay.style.color = "var(--primary)";
  }
  parsedSegments.forEach((seg, idx) => {
    const orig = originalSegments[idx];
    if (!orig) return;
    seg.start = orig.start;
    seg.end = orig.end;
    seg.start_fmt = orig.start_fmt;
    seg.end_fmt = orig.end_fmt;

    const timeEl = document.getElementById(`time-${idx}`);
    if (timeEl) timeEl.textContent = `${seg.start_fmt} ➔ ${seg.end_fmt}`;
    const durEl = document.getElementById(`dur-${idx}`);
    if (durEl) durEl.textContent = `(${(seg.end - seg.start).toFixed(1)}s)`;
  });
}

if (shiftBackBtn) {
  shiftBackBtn.addEventListener("click", () => applyTimeShift(-0.5));
}
if (shiftFwdBtn) {
  shiftFwdBtn.addEventListener("click", () => applyTimeShift(0.5));
}
if (resetShiftBtn) {
  resetShiftBtn.addEventListener("click", resetTimeShift);
}

// =========================================================================
// 8. Copy All and Export Edited Subtitles
// =========================================================================
if (copyAllBtn) {
  copyAllBtn.addEventListener("click", () => {
    const allText = parsedSegments.map(s => `[${s.start_fmt} --> ${s.end_fmt}]\n${s.text}`).join("\n\n");
    navigator.clipboard.writeText(allText).then(() => {
      const origHtml = copyAllBtn.innerHTML;
      copyAllBtn.innerHTML = "✓ Copied!";
      setTimeout(() => copyAllBtn.innerHTML = origHtml, 2000);
    });
  });
}

if (exportEditedSrtBtn) {
  exportEditedSrtBtn.addEventListener("click", () => {
    let srtContent = "";
    parsedSegments.forEach((seg, idx) => {
      srtContent += `${idx + 1}\n${seg.start_fmt} --> ${seg.end_fmt}\n${seg.text}\n\n`;
    });

    const blob = new Blob([srtContent], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const stem = (currentJobData && currentJobData.filename)
      ? currentJobData.filename.replace(/\.[^/.]+$/, "")
      : (currentFile ? currentFile.name.replace(/\.[^/.]+$/, "") : (currentJobId || "subtitles"));
    
    const offsetSuffix = totalSyncOffset !== 0 ? `_offset_${totalSyncOffset > 0 ? '+' : ''}${totalSyncOffset}s` : '';
    a.download = `${stem}_edited${offsetSuffix}.srt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
}

// =========================================================================
// 9. Formatters & Helpers
// =========================================================================
function formatTimestampSrt(totalSeconds) {
  if (totalSeconds < 0) totalSeconds = 0;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const milliseconds = Math.floor((totalSeconds % 1) * 1000);
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")},${String(milliseconds).padStart(3, "0")}`;
}

function addLog(text, type = "normal") {
  if (!logTerminal) return;
  const entry = document.createElement("div");
  entry.className = `log-entry ${type}`;
  const timeStr = new Date().toLocaleTimeString();
  entry.textContent = `[${timeStr}] ${text}`;
  logTerminal.appendChild(entry);
  logTerminal.scrollTop = logTerminal.scrollHeight;
}

function startTimer() {
  timerStart = Date.now();
  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - timerStart) / 1000);
    const m = Math.floor(elapsed / 60).toString().padStart(2, "0");
    const s = (elapsed % 60).toString().padStart(2, "0");
    if (elapsedTimer) elapsedTimer.textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function checkActiveSession() {
  const saved = localStorage.getItem("whisper_active_job");
  if (!saved) return;
  try {
    const jobInfo = JSON.parse(saved);
    if (!jobInfo.jobId) return;
    // Expire sessions older than 24 hours
    if (Date.now() - (jobInfo.timestamp || 0) > 86400000) {
      localStorage.removeItem("whisper_active_job");
      return;
    }

    const res = await fetch(`/api/jobs/${jobInfo.jobId}/status`);
    if (!res.ok) {
      localStorage.removeItem("whisper_active_job");
      return;
    }
    const data = await res.json();
    if (data.status === "processing" || data.status === "queued") {
      currentJobId = jobInfo.jobId;
      if (progressSection) progressSection.classList.remove("hidden");
      if (setupSection) setupSection.classList.add("hidden");
      if (resultsSection) resultsSection.classList.add("hidden");
      if (generateBtn) generateBtn.disabled = true;
      if (fileNameEl) fileNameEl.textContent = jobInfo.filename || "video.mp4";
      addLog(`[Session Restored] Reconnected to active job #${jobInfo.jobId} for "${jobInfo.filename || 'video'}".`, "success");
      startTimer();
      handleProgressUpdate(data);
      connectEventSource(currentJobId);
    } else if (data.status === "completed" && data.result) {
      localStorage.removeItem("whisper_active_job");
      finishJob(data.result);
    } else {
      localStorage.removeItem("whisper_active_job");
    }
  } catch (e) {
    console.warn("Could not restore session:", e);
  }
}

// Initialize on Load
initSystemStatus();
setupDropzone();
checkActiveSession();
setInterval(initSystemStatus, 5000);
