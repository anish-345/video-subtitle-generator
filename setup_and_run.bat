@echo off
title Whisper Subtitle Studio - Rust & GPU Engine
cd /d "%~dp0"

echo ===================================================
echo     WHISPER SUBTITLE STUDIO - RUST & GPU ENGINE
echo ===================================================

if not exist ".venv\Scripts\python.exe" (
    echo Setting up Python virtual environment...
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        py -3 -m venv .venv
    ) else (
        where python >nul 2>&1
        if %errorlevel% equ 0 (
            python -m venv .venv
        ) else (
            echo [ERROR] Python 3.10+ was not found in PATH.
            echo Please install Python and ensure 'python' or 'py' is accessible in your terminal.
            pause
            exit /b 1
        )
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

if not exist ".\rust-gateway\target\release\rust-gateway.exe" (
    echo Building Rust Concurrency Gateway...
    where cargo >nul 2>&1
    if %errorlevel% equ 0 (
        cargo build --release --manifest-path "rust-gateway\Cargo.toml"
    ) else (
        echo [ERROR] Rust / Cargo not found in PATH.
        echo Please install Rust from https://rustup.rs to build the Rust Gateway.
        pause
        exit /b 1
    )
)

echo [1/2] Starting GPU Whisper Worker on port 8001...
start "Whisper GPU Worker (8001)" /B ".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8001

timeout /t 2 /nobreak >nul

echo [2/2] Starting Rust Concurrency Core on port 8000...
echo Opening your browser to http://localhost:8000 ...
start http://localhost:8000

".\rust-gateway\target\release\rust-gateway.exe"
pause
