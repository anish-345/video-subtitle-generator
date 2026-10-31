# Launch script for Whisper Subtitle Studio (Rust + GPU Platform)

$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location -Path $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   WHISPER SUBTITLE STUDIO (Rust + GPU/CPU Engine)      " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Ensure ports 8000 & 8001 are free from any previous session
try {
    $occupied = Get-NetTCPConnection -LocalPort 8000, 8001 -State Listen -ErrorAction SilentlyContinue
    if ($occupied) {
        $pids = $occupied | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($procId in $pids) {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 500
    }
} catch {}

# 1. Check Python Venv
$PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "Setting up Python virtual environment..." -ForegroundColor Yellow
    $SysPython = $null
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        $SysPython = "python"
    } elseif (Get-Command "py" -ErrorAction SilentlyContinue) {
        $SysPython = "py"
    } elseif (Get-Command "python3" -ErrorAction SilentlyContinue) {
        $SysPython = "python3"
    } else {
        Write-Error "Python 3.10+ was not found in PATH. Please install Python and ensure it is in your PATH."
        exit 1
    }
    & $SysPython -m venv .venv
    & $PythonExe -m pip install --upgrade pip
    & $PythonExe -m pip install -r requirements.txt
}

# 2. Ensure Rust Gateway binary is built
$RustExe = Join-Path $PSScriptRoot "rust-gateway\target\release\rust-gateway.exe"
if (-not (Test-Path $RustExe)) {
    Write-Host "Building Rust Concurrency Core (release mode)..." -ForegroundColor Cyan
    if (Get-Command "cargo" -ErrorAction SilentlyContinue) {
        cargo build --release --manifest-path (Join-Path $PSScriptRoot "rust-gateway\Cargo.toml")
    } else {
        Write-Error "Cargo was not found in PATH. Please install Rust from https://rustup.rs to run the Rust Concurrency Core."
        exit 1
    }
}

# 3. Detect Local/Wi-Fi IP
$LocalIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "*Wi-Fi*" -ErrorAction SilentlyContinue | Select-Object -First 1).IPAddress
if (-not $LocalIP) {
    $LocalIP = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } | Select-Object -First 1).IPAddress
}
if (-not $LocalIP) { $LocalIP = "127.0.0.1" }

# Optional Cloudflare Tunnel Integration
$TunnelProcess = $null
$TunnelTokenFile = Join-Path $PSScriptRoot "tunnel_token.txt"
if (Test-Path $TunnelTokenFile) {
    $Token = (Get-Content $TunnelTokenFile -Raw).Trim()
    if ($Token) {
        Write-Host "Starting Cloudflare Custom Domain Tunnel..." -ForegroundColor Magenta
        $TunnelProcess = Start-Process -FilePath "cloudflared.exe" -ArgumentList "tunnel run --token $Token" -PassThru -NoNewWindow
    }
}

Write-Host "--------------------------------------------------------" -ForegroundColor Yellow
Write-Host "Access URLs:" -ForegroundColor Yellow
Write-Host "  - Local PC:       http://localhost:8000" -ForegroundColor White
if ($LocalIP -ne "127.0.0.1") {
    Write-Host "  - Local Network:  http://${LocalIP}:8000" -ForegroundColor White
}
if ($TunnelProcess) {
    Write-Host "  - Cloudflare:     Custom Domain Tunnel Active" -ForegroundColor Magenta
}
Write-Host "--------------------------------------------------------`n" -ForegroundColor Yellow

# Open browser
Start-Process "http://localhost:8000"

# Start GPU Worker on port 8001
Write-Host "[1/2] Starting GPU Whisper Worker on port 8001..." -ForegroundColor Green
$WorkerProcess = Start-Process -FilePath $PythonExe -ArgumentList "-m uvicorn backend.main:app --host 127.0.0.1 --port 8001" -PassThru -NoNewWindow
Start-Sleep -Seconds 2

# Run Rust Gateway on port 8000
Write-Host "[2/2] Rust Concurrency Core active on port 8000. Press Ctrl+C anytime to stop.`n" -ForegroundColor Green
try {
    & $RustExe
} finally {
    Write-Host "`nShutting down GPU Worker..." -ForegroundColor Gray
    Stop-Process -Id $WorkerProcess.Id -ErrorAction SilentlyContinue
    if ($TunnelProcess) {
        Write-Host "Shutting down Cloudflare Tunnel..." -ForegroundColor Gray
        Stop-Process -Id $TunnelProcess.Id -ErrorAction SilentlyContinue
    }
}
