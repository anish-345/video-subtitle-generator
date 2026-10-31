import os
import shutil
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
SUBTITLES_DIR = DATA_DIR / "subtitles"
TEMP_DIR = DATA_DIR / "temp"
OUTPUTS_DIR = DATA_DIR / "outputs"

for directory in [UPLOADS_DIR, SUBTITLES_DIR, TEMP_DIR, OUTPUTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def find_ffmpeg() -> str:
    """Find FFmpeg binary across environment variables, PATH, local project folders, or system paths."""
    # 1. Custom environment variable override
    env_ffmpeg = os.environ.get("FFMPEG_PATH")
    if env_ffmpeg and os.path.isfile(env_ffmpeg):
        return env_ffmpeg

    # 2. Check system PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    # 3. Project-local directories
    bin_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    local_candidates = [
        BASE_DIR / "bin" / bin_name,
        BASE_DIR / "ffmpeg" / "bin" / bin_name,
        BASE_DIR / "ffmpeg" / bin_name,
    ]
    for candidate in local_candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    # 4. Standard platform-specific directories
    if sys.platform == "win32":
        win_candidates = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "ffmpeg" / "bin" / "ffmpeg.exe",
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"),
            Path(os.environ.get("USERPROFILE", "")) / "scoop" / "shims" / "ffmpeg.exe",
        ]
        for candidate in win_candidates:
            if candidate.is_file():
                return str(candidate.resolve())
    elif sys.platform in ("linux", "darwin"):
        unix_candidates = [
            Path("/usr/bin/ffmpeg"),
            Path("/usr/local/bin/ffmpeg"),
            Path("/opt/homebrew/bin/ffmpeg"),
        ]
        for candidate in unix_candidates:
            if candidate.is_file():
                return str(candidate.resolve())

    # Fallback to binary name
    return "ffmpeg"

FFMPEG_PATH = find_ffmpeg()

def find_ffprobe() -> str:
    """Find FFprobe binary matching FFmpeg or in PATH."""
    env_ffprobe = os.environ.get("FFPROBE_PATH")
    if env_ffprobe and os.path.isfile(env_ffprobe):
        return env_ffprobe

    probe_name = "ffprobe.exe" if sys.platform == "win32" else "ffprobe"

    # Look alongside ffmpeg binary
    if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
        probe = Path(FFMPEG_PATH).parent / probe_name
        if probe.is_file():
            return str(probe.resolve())

    found = shutil.which("ffprobe")
    if found:
        return found

    return "ffprobe"

FFPROBE_PATH = find_ffprobe()

def cleanup_file(file_path: str | Path | None) -> bool:
    """Safely delete a temporary or intermediate file."""
    if not file_path:
        return False
    try:
        p = Path(file_path)
        if p.exists() and p.is_file():
            p.unlink()
            return True
    except Exception as e:
        print(f"[Cleanup Warning] Could not remove {file_path}: {e}")
    return False

def cleanup_directory(dir_path: str | Path | None) -> bool:
    """Safely delete a temporary directory."""
    if not dir_path:
        return False
    try:
        p = Path(dir_path)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            return True
    except Exception as e:
        print(f"[Cleanup Warning] Could not remove dir {dir_path}: {e}")
    return False

def purge_expired_temp_files(max_age_seconds: int = 3600):
    """Purge temporary intermediate files older than max_age_seconds."""
    now = time.time()
    for item in TEMP_DIR.glob("*"):
        try:
            if item.stat().st_mtime < now - max_age_seconds:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

def purge_expired_subtitles(max_age_seconds: int = 86400):
    """Purge subtitle files older than max_age_seconds (default 24 hours)."""
    now = time.time()
    for item in SUBTITLES_DIR.glob("*"):
        try:
            if item.stat().st_mtime < now - max_age_seconds:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

def purge_expired_outputs(max_age_seconds: int = 86400):
    """Purge output files older than max_age_seconds (default 24 hours)."""
    now = time.time()
    for item in OUTPUTS_DIR.glob("*"):
        try:
            if item.stat().st_mtime < now - max_age_seconds:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

