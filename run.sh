#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "   WHISPER SUBTITLE STUDIO (Rust + GPU/CPU Engine)      "
echo "========================================================"

# 1. Setup Python Virtual Environment
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo "Setting up Python virtual environment..."
    PYTHON_CMD=""
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "[ERROR] Python 3.10+ was not found. Please install Python."
        exit 1
    fi
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    "$VENV_DIR/bin/python" -m pip install -r requirements.txt
fi

PYTHON_EXE="$VENV_DIR/bin/python"

# 2. Ensure Rust Gateway binary is built
RUST_EXE="$SCRIPT_DIR/rust-gateway/target/release/rust-gateway"
if [ ! -f "$RUST_EXE" ]; then
    echo "Building Rust Concurrency Gateway in release mode..."
    if command -v cargo &> /dev/null; then
        cargo build --release --manifest-path "$SCRIPT_DIR/rust-gateway/Cargo.toml"
    else
        echo "[ERROR] Cargo was not found in PATH. Please install Rust from https://rustup.rs"
        exit 1
    fi
fi

# Detect Local IP
LOCAL_IP="127.0.0.1"
if command -v hostname &> /dev/null; then
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}') || LOCAL_IP="127.0.0.1"
fi

echo "--------------------------------------------------------"
echo "Access URLs:"
echo "  - Local:         http://localhost:8000"
if [ "$LOCAL_IP" != "127.0.0.1" ] && [ -n "$LOCAL_IP" ]; then
    echo "  - Local Network: http://${LOCAL_IP}:8000"
fi
echo "--------------------------------------------------------"

# Open browser if possible
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8000" &> /dev/null &
elif command -v open &> /dev/null; then
    open "http://localhost:8000" &> /dev/null &
fi

echo "[1/2] Starting GPU Whisper Worker on port 8001..."
"$PYTHON_EXE" -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 &
WORKER_PID=$!

cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$WORKER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2
echo "[2/2] Rust Concurrency Core active on port 8000. Press Ctrl+C anytime to stop."
"$RUST_EXE"

