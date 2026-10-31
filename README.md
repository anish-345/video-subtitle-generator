# 🎬 Whisper GPU Subtitle Studio

A modern, high-performance web application built as a **single unified system** combining a high-concurrency **Rust Gateway** (Port 8000) with a hardware-accelerated **Python GPU Inference Worker** (Port 8001). Generates high-accuracy subtitles for videos using **Whisper** and **Alibaba SenseVoice** accelerated by GPU (CUDA / Tensor Cores) or CPU, with real-time in-browser caption playback, studio audio denoising, and multi-format exports (`.srt`, `.vtt`, `.txt`, `.mp3`).

---

## 🏛️ System Architecture

The project operates as a unified dual-engine pipeline:
1. **Rust Concurrency Core (Port 8000)**: Built with Tokio + Axum. Serves the web UI, streams SSE progress events, queues concurrent client video uploads, and serializes GPU access to eliminate out-of-memory errors.
2. **CUDA GPU Inference Worker (Port 8001)**: Built with FastAPI + `faster-whisper` + CTranslate2. Executes batch tensor-core speech recognition and audio normalization.

Both services are automatically managed and started together by the unified launch scripts (`run.ps1`, `setup_and_run.bat`, or `run.sh`).

---

## ✨ Features

- 🦀 **Rust Concurrency Core**: Tokio + Axum gateway that manages concurrent client uploads, GPU task serialization to prevent OOM errors, and real-time streaming progress.
- ⚡ **Hardware Acceleration**: Automatically detects NVIDIA GPUs (CUDA / Tensor Cores via `faster-whisper` and CTranslate2) with `float16` compute type, falling back smoothly to CPU `int8` on systems without a dedicated GPU.
- 🎙️ **Acoustic Noise Cancellation & Voice Clarity**:
  - Automatically filters extracted audio with adaptive FFT spectral denoising (`afftdn`), sub-bass rumble cuts (<80Hz), and high-frequency hiss removal (>12kHz).
  - Enhances human vocal formants (2.8kHz presence boost) and applies Dynamic Audio Normalization (`dynaudnorm`) to boost faint or whispering voices without loud distortion.
  - Three tuned presets: *Studio Voice Clarity* (recommended), *Aggressive Noise Removal* (fans, AC, traffic), and *Voice Boost & Leveling* (quiet/distant microphones).
  - Includes a one-click **Download Clean Audio (.MP3)** export in the Subtitle Studio workspace.
- 🧹 **Automatic Cleanup & Storage Management**:
  - Extracted 16kHz WAV intermediate audio files are automatically purged once transcription completes.
  - One-click "Clean Media" button in the UI lets you purge the uploaded video file anytime to free disk storage while keeping generated subtitles intact.
- 🎯 **Supported Speech Recognition Models**:
  - `base` (~140MB) - Default recommended, balanced speed and accuracy.
  - `sensevoice-small` (~1.1GB) - Alibaba SenseVoice with millisecond-precision audio timestamping.
  - `tiny` (~75MB) - Ultra fast.
  - `small` (~460MB) - High accuracy.
  - `medium` (~1.5GB) - Studio quality.
  - `large-v3-turbo` (~1.6GB) - SOTA deep learning speech recognition.
  - `large-v3` (~3GB) - Maximum multilingual accuracy.
- 🌍 **Multilingual & Translation**:
  - Auto-detects spoken languages (English, Hindi, Spanish, French, German, Japanese, Chinese, and 90+ others).
  - Built-in translation mode: transcribes foreign speech directly into English subtitles.
- 📺 **Interactive In-Browser Video Player**:
  - Live synchronized subtitle playback.
  - Interactive transcript editor: edit subtitle cues directly and export customized `.srt` files.
  - Real-time cue synchronization offset controls (shift forward / back by 100ms increments).
- 💾 **Multi-Format Export**:
  - SubRip Subtitle (`.srt`)
  - WebVTT (`.vtt`)
  - Plain Text Transcript (`.txt`)
  - Studio Denoised Audio (`.mp3`)

---

## 📋 Prerequisites

1. **Python 3.10+**: Ensure Python is installed and available in your `PATH`.
2. **FFmpeg**: Required for audio extraction and conversion.
   - **Windows**: `winget install Gyan.FFmpeg` or `choco install ffmpeg`, or place `ffmpeg.exe` in `bin/` or `PATH`.
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo pacman -S ffmpeg` (Arch)
3. **Rust & Cargo**: Required to compile and run the high-concurrency Rust Gateway (`https://rustup.rs`).
4. **NVIDIA GPU & CUDA Drivers** *(Optional, for hardware acceleration)*:
   - For NVIDIA GPU acceleration, ensure current NVIDIA drivers and CUDA are installed.

---

## 🚀 Quick Start

### Windows

Run via PowerShell:
```powershell
.\run.ps1
```
Or double-click:
```bat
start.bat
```
*(Or `setup_and_run.bat`)*

### Linux & macOS

Make executable and launch:
```bash
chmod +x run.sh
./run.sh
```

### Manual Launch

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Start the application:
   ```bash
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

---

## 🧪 Testing

Run automated tests to verify hardware acceleration, audio extraction, and transcription:
```bash
python test_transcription.py
```

Test concurrency and multi-client queue processing:
```bash
python test_concurrency.py
```

---

## 🌐 Local Network Access

When running, the console displays both your local address (`http://localhost:8000`) and your local network / Wi-Fi address (e.g. `http://192.168.x.x:8000`), allowing phones, tablets, or other PCs on your local network to use the subtitle generator directly.
