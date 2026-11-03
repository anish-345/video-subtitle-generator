import os
import subprocess
import time
import wave
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import requests

from backend.config import FFMPEG_PATH, TEMP_DIR, MODELS_DIR

ENHANCER_MODELS_DIR = MODELS_DIR / "audio_enhancer"
ENHANCER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

GTCRN_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx"
UVR_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_9482.onnx"

GTCRN_PATH = ENHANCER_MODELS_DIR / "gtcrn_simple.onnx"
UVR_PATH = ENHANCER_MODELS_DIR / "UVR_MDXNET_9482.onnx"


def ensure_model_file(model_path: Path, download_url: str, name: str, progress_cb: Optional[Callable[[str], None]] = None) -> bool:
    """Ensure ONNX model file exists; auto-download from official release if missing."""
    if model_path.exists() and model_path.stat().st_size > 10000:
        return True

    if progress_cb:
        progress_cb(f"Downloading deep learning AI model: {name}...")
    print(f"[DeepEnhancer] Auto-downloading {name} from {download_url}...")

    tmp_path = model_path.with_suffix(".tmp")
    try:
        response = requests.get(download_url, stream=True, timeout=90)
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        if tmp_path.exists() and tmp_path.stat().st_size > 10000:
            tmp_path.replace(model_path)
            print(f"[DeepEnhancer] {name} successfully downloaded to {model_path}")
            return True
    except Exception as e:
        print(f"[DeepEnhancer] Failed to download {name}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False
    return False


def load_wav_file(wav_path: str | Path):
    """Load WAV audio into numpy float32 array in range [-1.0, 1.0]."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

    if ch > 1:
        audio = audio.reshape(-1, ch).T  # shape: (channels, samples)
    else:
        audio = audio.reshape(1, -1)
    return audio, sr


def save_wav_file(wav_path: str | Path, samples: np.ndarray, sr: int):
    """Save numpy float32 array to 16-bit PCM WAV."""
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767.0).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        if int_samples.ndim == 1:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int_samples.tobytes())
        else:
            ch, _ = int_samples.shape
            wf.setnchannels(ch)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            interleaved = int_samples.T.flatten()
            wf.writeframes(interleaved.tobytes())


class DeepAudioEnhancer:
    """
    SOTA Deep Learning Audio Enhancement & Background Music Removal Engine:
    - UVR MDX-NET: Isolates speech / vocals, stripping out background music, songs, and instruments.
    - GTCRN: Deep complex recurrent convolutional neural network speech denoiser.
    """

    def __init__(self):
        self._uvr_separator = None
        self._gtcrn_denoiser = None
        self._sherpa_available = None

    def is_available(self) -> bool:
        if self._sherpa_available is not None:
            return self._sherpa_available
        try:
            import sherpa_onnx
            self._sherpa_available = True
        except ImportError:
            self._sherpa_available = False
        return self._sherpa_available

    def _get_provider(self) -> str:
        # Prefer CPU or CUDA based on environment
        return "cpu"

    def get_uvr_separator(self, progress_cb: Optional[Callable[[str], None]] = None):
        if self._uvr_separator is not None:
            return self._uvr_separator

        if not self.is_available():
            return None

        if not ensure_model_file(UVR_PATH, UVR_URL, "UVR MDX-Net Vocal Separator", progress_cb):
            return None

        import sherpa_onnx
        try:
            cfg = sherpa_onnx.OfflineSourceSeparationConfig(
                model=sherpa_onnx.OfflineSourceSeparationModelConfig(
                    uvr=sherpa_onnx.OfflineSourceSeparationUvrModelConfig(model=str(UVR_PATH)),
                    provider=self._get_provider(),
                    num_threads=2
                )
            )
            self._uvr_separator = sherpa_onnx.OfflineSourceSeparation(cfg)
            print("[DeepEnhancer] UVR MDX-Net Vocal Separator initialized successfully.")
            return self._uvr_separator
        except Exception as e:
            print(f"[DeepEnhancer] Failed initializing UVR: {e}")
            return None

    def get_gtcrn_denoiser(self, progress_cb: Optional[Callable[[str], None]] = None):
        if self._gtcrn_denoiser is not None:
            return self._gtcrn_denoiser

        if not self.is_available():
            return None

        if not ensure_model_file(GTCRN_PATH, GTCRN_URL, "GTCRN Neural Speech Denoiser", progress_cb):
            return None

        import sherpa_onnx
        try:
            cfg = sherpa_onnx.OfflineSpeechDenoiserConfig(
                model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                    gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(GTCRN_PATH)),
                    provider=self._get_provider(),
                    num_threads=2
                )
            )
            self._gtcrn_denoiser = sherpa_onnx.OfflineSpeechDenoiser(cfg)
            print("[DeepEnhancer] GTCRN Neural Speech Denoiser initialized successfully.")
            return self._gtcrn_denoiser
        except Exception as e:
            print(f"[DeepEnhancer] Failed initializing GTCRN: {e}")
            return None

    def separate_music_and_vocals(self, input_wav_44k: Path, output_vocals_wav: Path, progress_cb: Optional[Callable[[str], None]] = None) -> bool:
        """Run UVR MDX-Net to separate speech/vocals from background music and instruments."""
        sep = self.get_uvr_separator(progress_cb)
        if not sep:
            return False

        if progress_cb:
            progress_cb("Separating speech from background music (UVR MDX-Net AI)...")

        audio_stereo, sr = load_wav_file(input_wav_44k)
        if audio_stereo.shape[0] == 1:
            # Duplicate mono to stereo for UVR
            audio_stereo = np.vstack([audio_stereo, audio_stereo])

        total_samples = audio_stereo.shape[1]
        chunk_samples = sr * 30  # 30-second processing chunks to preserve memory
        total_chunks = (total_samples + chunk_samples - 1) // chunk_samples

        vocal_parts = []
        for idx, start in enumerate(range(0, total_samples, chunk_samples)):
            end = min(start + chunk_samples, total_samples)
            if progress_cb:
                pct = int((idx / max(1, total_chunks)) * 100)
                progress_cb(f"Separating vocals & music (UVR AI): chunk {idx+1}/{total_chunks} ({pct}%)...")

            chunk = np.ascontiguousarray(audio_stereo[:, start:end])
            if chunk.shape[1] < sr:
                pad_len = sr - chunk.shape[1]
                padded = np.ascontiguousarray(np.pad(chunk, ((0, 0), (0, pad_len))))
                out = sep.process(sr, padded)
                vocal_chunk = np.array(out.stems[0].data)[:, :end - start]
            else:
                out = sep.process(sr, chunk)
                vocal_chunk = np.array(out.stems[0].data)
            vocal_parts.append(vocal_chunk)

        vocals_stereo = np.concatenate(vocal_parts, axis=1)
        # Convert isolated vocals to mono
        vocals_mono = np.mean(vocals_stereo, axis=0)
        save_wav_file(output_vocals_wav, vocals_mono, sr)
        return True

    def denoise_audio_gtcrn(self, input_wav_16k: Path, output_denoised_wav: Path, progress_cb: Optional[Callable[[str], None]] = None) -> bool:
        """Run GTCRN neural network speech denoiser."""
        denoiser = self.get_gtcrn_denoiser(progress_cb)
        if not denoiser:
            return False

        if progress_cb:
            progress_cb("Applying deep learning speech denoising (GTCRN AI)...")

        audio_mono, sr = load_wav_file(input_wav_16k)
        audio_flat = audio_mono.flatten()
        total_samples = len(audio_flat)
        chunk_samples = sr * 30
        total_chunks = (total_samples + chunk_samples - 1) // chunk_samples

        parts = []
        for idx, start in enumerate(range(0, total_samples, chunk_samples)):
            end = min(start + chunk_samples, total_samples)
            if progress_cb:
                pct = int((idx / max(1, total_chunks)) * 100)
                progress_cb(f"Applying neural speech denoising (GTCRN AI): chunk {idx+1}/{total_chunks} ({pct}%)...")

            chunk = np.ascontiguousarray(audio_flat[start:end])
            out = denoiser.run(chunk, sr)
            parts.append(np.array(out.samples, dtype=np.float32))

        denoised = np.concatenate(parts)
        save_wav_file(output_denoised_wav, denoised, sr)
        return True

    def process_enhancement(
        self,
        video_path: Path,
        output_16k_audio: Path,
        mode: str = "deep_clean",
        clean_mp3_path: Optional[Path] = None,
        progress_cb: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Orchestrates audio extraction and deep learning enhancement:
        - 'deep_clean': Music Removal (UVR MDX-Net) + GTCRN Speech Denoising
        - 'music_removal': Music & Song Removal only (UVR MDX-Net)
        - 'denoise_ai': GTCRN Neural Speech Denoising only
        """
        job_tag = f"dl_{int(time.time() * 1000)}"

        try:
            if mode in ("deep_clean", "music_removal"):
                # 1. Extract 44.1kHz stereo WAV for UVR
                raw_44k = TEMP_DIR / f"{job_tag}_raw_44k.wav"
                cmd_ext = [
                    FFMPEG_PATH, "-y", "-threads", "0",
                    "-i", str(video_path), "-vn",
                    "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                    str(raw_44k)
                ]
                subprocess.run(cmd_ext, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

                # 2. Run UVR to separate music from vocal speech
                vocals_44k = TEMP_DIR / f"{job_tag}_vocals_44k.wav"
                sep_ok = self.separate_music_and_vocals(raw_44k, vocals_44k, progress_cb)
                if raw_44k.exists():
                    raw_44k.unlink()

                if not sep_ok:
                    print("[DeepEnhancer] UVR failed, falling back to standard extraction.")
                    return False

                # 3. Resample vocals to 16kHz mono
                vocals_16k = TEMP_DIR / f"{job_tag}_vocals_16k.wav"
                cmd_resample = [
                    FFMPEG_PATH, "-y", "-threads", "0",
                    "-i", str(vocals_44k),
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(vocals_16k)
                ]
                subprocess.run(cmd_resample, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                if vocals_44k.exists():
                    vocals_44k.unlink()

                # 4. If deep_clean is requested, also run GTCRN denoiser on the vocal stem
                if mode == "deep_clean":
                    denoised_16k = TEMP_DIR / f"{job_tag}_final_16k.wav"
                    den_ok = self.denoise_audio_gtcrn(vocals_16k, denoised_16k, progress_cb)
                    if den_ok and denoised_16k.exists():
                        denoised_16k.replace(output_16k_audio)
                    else:
                        vocals_16k.replace(output_16k_audio)
                else:
                    vocals_16k.replace(output_16k_audio)

                if vocals_16k.exists():
                    vocals_16k.unlink()

            elif mode == "denoise_ai":
                # 1. Extract 16kHz mono WAV
                raw_16k = TEMP_DIR / f"{job_tag}_raw_16k.wav"
                cmd_ext = [
                    FFMPEG_PATH, "-y", "-threads", "0",
                    "-i", str(video_path), "-vn",
                    "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(raw_16k)
                ]
                subprocess.run(cmd_ext, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

                # 2. Run GTCRN denoiser
                den_ok = self.denoise_audio_gtcrn(raw_16k, output_16k_audio, progress_cb)
                if raw_16k.exists():
                    raw_16k.unlink()
                if not den_ok:
                    return False

            # Generate high-quality clean MP3 for user preview / download if requested
            if clean_mp3_path and output_16k_audio.exists():
                cmd_mp3 = [
                    FFMPEG_PATH, "-y", "-threads", "0",
                    "-i", str(output_16k_audio),
                    "-c:a", "libmp3lame", "-b:a", "192k",
                    str(clean_mp3_path)
                ]
                subprocess.run(cmd_mp3, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            return True
        except Exception as e:
            print(f"[DeepEnhancer Error] {e}")
            return False


deep_audio_enhancer = DeepAudioEnhancer()

