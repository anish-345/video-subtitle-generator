@echo off
setlocal enabledelayedexpansion

echo ==========================================================
echo    WHISPER SUBTITLE STUDIO - 1-CLICK CUDA ACCELERATION
echo ==========================================================
echo.

set "PY_EXE="
if exist "%~dp0python\python.exe" (
    set "PY_EXE=%~dp0python\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    for /f "tokens=*" %%i in ('where python 2^>nul') do (
        if not defined PY_EXE set "PY_EXE=%%i"
    )
)

if not defined PY_EXE (
    echo [ERROR] No Python runtime found. Please install Python 3.10+ or run setup_and_run.bat first.
    pause
    exit /b 1
)

echo [*] Using Python: %PY_EXE%
echo [*] Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [NOTICE] nvidia-smi command did not find an NVIDIA GPU.
    echo If your PC has an NVIDIA card, ensure latest NVIDIA drivers are installed from:
    echo https://www.nvidia.com/Download/index.aspx
    echo.
) else (
    echo [OK] Physical NVIDIA GPU detected!
)

echo.
echo [*] Installing NVIDIA cuBLAS 12 and cuDNN 9 acceleration packages...
"%PY_EXE%" -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 --no-warn-script-location

echo.
echo [*] Verifying CUDA acceleration with CTranslate2...
"%PY_EXE%" -c "import ctranslate2; count = ctranslate2.get_cuda_device_count(); print(f'[STATUS] CUDA devices available to CTranslate2: {count}'); exit(0 if count > 0 else 1)"
if %errorlevel% equ 0 (
    echo.
    echo ==========================================================
    echo [SUCCESS] CUDA Acceleration is ACTIVE! (20x-50x speedup)
    echo ==========================================================
) else (
    echo.
    echo [NOTICE] Packages installed. If CUDA device count is still 0,
    echo please restart Whisper Studio so Windows reloads the DLL path.
)

echo.
pause
