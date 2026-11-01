# Windows Packaging Script for Whisper Subtitle Studio
# Creates portable distribution and standalone Inno Setup installer

param(
    [switch]$SkipPythonEmbed = $false
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent -Path $ScriptRoot
Set-Location $ProjectRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   PACKAGING WHISPER SUBTITLE STUDIO (WINDOWS x64)        " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$PackageDir = Join-Path $ProjectRoot "package"
$DistDir = Join-Path $ProjectRoot "dist"

if (Test-Path $PackageDir) { Remove-Item $PackageDir -Recurse -Force }
if (-not (Test-Path $DistDir)) { New-Item -ItemType Directory -Path $DistDir -Force | Out-Null }
New-Item -ItemType Directory -Path $PackageDir -Force | Out-Null

# 1. Build Rust Gateway
Write-Host "`n[1/6] Compiling Rust Concurrency Gateway (Release Mode)..." -ForegroundColor Green
cargo build --release --manifest-path (Join-Path $ProjectRoot "rust-gateway\Cargo.toml")
$BuiltRustExe = Join-Path $ProjectRoot "rust-gateway\target\release\rust-gateway.exe"
Copy-Item $BuiltRustExe (Join-Path $PackageDir "rust-gateway.exe")

# 2. Copy Project Assets
Write-Host "`n[2/6] Copying application assets..." -ForegroundColor Green
Copy-Item (Join-Path $ProjectRoot "backend") (Join-Path $PackageDir "backend") -Recurse
Copy-Item (Join-Path $ProjectRoot "frontend") (Join-Path $PackageDir "frontend") -Recurse
Copy-Item (Join-Path $ProjectRoot "requirements.txt") (Join-Path $PackageDir "requirements.txt")
Copy-Item (Join-Path $ProjectRoot "README.md") (Join-Path $PackageDir "README.md")
Copy-Item (Join-Path $ProjectRoot "start.bat") (Join-Path $PackageDir "start.bat")
Copy-Item (Join-Path $ProjectRoot "setup_and_run.bat") (Join-Path $PackageDir "setup_and_run.bat")
Copy-Item (Join-Path $ProjectRoot "setup_cuda.bat") (Join-Path $PackageDir "setup_cuda.bat")
Copy-Item (Join-Path $ProjectRoot "run.ps1") (Join-Path $PackageDir "run.ps1")

# Clean __pycache__ from package
Get-ChildItem -Path $PackageDir -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Ensure data directories exist with .gitkeep
$DataDirs = @("uploads", "outputs", "subtitles", "temp", "models")
foreach ($d in $DataDirs) {
    $dirPath = Join-Path $PackageDir "data\$d"
    New-Item -ItemType Directory -Path $dirPath -Force | Out-Null
    Set-Content -Path (Join-Path $dirPath ".gitkeep") -Value ""
}

# 3. Bundle FFmpeg and FFprobe
Write-Host "`n[3/6] Bundling FFmpeg & FFprobe binaries..." -ForegroundColor Green
$BinDir = Join-Path $PackageDir "bin"
New-Item -ItemType Directory -Path $BinDir -Force | Out-Null

$LocalFFmpeg = Join-Path $ProjectRoot "bin\ffmpeg.exe"
$LocalFFprobe = Join-Path $ProjectRoot "bin\ffprobe.exe"

if ((Test-Path $LocalFFmpeg) -and (Test-Path $LocalFFprobe)) {
    Copy-Item $LocalFFmpeg (Join-Path $BinDir "ffmpeg.exe")
    Copy-Item $LocalFFprobe (Join-Path $BinDir "ffprobe.exe")
} else {
    Write-Host "Downloading standalone FFmpeg release from GitHub..." -ForegroundColor Yellow
    $FfmpegZip = Join-Path $env:TEMP "ffmpeg-release.zip"
    $FfmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $FfmpegUrl -OutFile $FfmpegZip
        $ExtractDir = Join-Path $env:TEMP "ffmpeg-extracted"
        if (Test-Path $ExtractDir) { Remove-Item $ExtractDir -Recurse -Force }
        Expand-Archive -Path $FfmpegZip -DestinationPath $ExtractDir -Force
        
        $FoundFfmpeg = Get-ChildItem -Path $ExtractDir -Recurse -Filter "ffmpeg.exe" | Select-Object -First 1
        $FoundFfprobe = Get-ChildItem -Path $ExtractDir -Recurse -Filter "ffprobe.exe" | Select-Object -First 1
        
        if ($FoundFfmpeg -and $FoundFfprobe) {
            Copy-Item $FoundFfmpeg.FullName (Join-Path $BinDir "ffmpeg.exe")
            Copy-Item $FoundFfprobe.FullName (Join-Path $BinDir "ffprobe.exe")
            Write-Host "✓ FFmpeg & FFprobe bundled successfully." -ForegroundColor Green
        }
        Remove-Item $FfmpegZip -Force -ErrorAction SilentlyContinue
        Remove-Item $ExtractDir -Recurse -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Host "Notice: Online FFmpeg download skipped or failed ($($_.Exception.Message)). Will use system FFmpeg." -ForegroundColor Yellow
    }
}

# 4. Bundle Portable Python Runtime (Embeddable Package)
Write-Host "`n[4/6] Setting up Portable Python Runtime..." -ForegroundColor Green
$PackagePythonDir = Join-Path $PackageDir "python"
if (-not $SkipPythonEmbed) {
    Write-Host "Downloading Python 3.11 Embeddable Package..." -ForegroundColor Yellow
    $PyEmbedZip = Join-Path $env:TEMP "python-embed.zip"
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" -OutFile $PyEmbedZip
    New-Item -ItemType Directory -Path $PackagePythonDir -Force | Out-Null
    Expand-Archive -Path $PyEmbedZip -DestinationPath $PackagePythonDir -Force

    $PthFile = Join-Path $PackagePythonDir "python311._pth"
    $PthLines = Get-Content $PthFile
    $PthLines = $PthLines -replace "#import site", "import site"
    $PthLines += "`nLib\site-packages`n..`n."
    Set-Content -Path $PthFile -Value $PthLines

    $GetPip = Join-Path $env:TEMP "get-pip.py"
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip
    & (Join-Path $PackagePythonDir "python.exe") $GetPip --no-warn-script-location

    Write-Host "Installing requirements into portable Python..." -ForegroundColor Cyan
    & (Join-Path $PackagePythonDir "python.exe") -m pip install -r (Join-Path $ProjectRoot "requirements.txt") --no-warn-script-location
    Remove-Item $PyEmbedZip -Force -ErrorAction SilentlyContinue
    Remove-Item $GetPip -Force -ErrorAction SilentlyContinue
}

# 5. Create Portable Zip
Write-Host "`n[5/6] Creating Portable Distribution Zip..." -ForegroundColor Green
$ZipFile = Join-Path $DistDir "WhisperStudio-Portable-x64.zip"
if (Test-Path $ZipFile) { Remove-Item $ZipFile -Force }
Compress-Archive -Path "$PackageDir\*" -DestinationPath $ZipFile -CompressionLevel Optimal
Write-Host "✓ Created: $ZipFile" -ForegroundColor Green

# 6. Build Inno Setup Installer (if ISCC is available)
Write-Host "`n[6/6] Checking for Inno Setup Compiler..." -ForegroundColor Green
$Iscc = $null
if (Get-Command "iscc" -ErrorAction SilentlyContinue) {
    $Iscc = "iscc"
} elseif (Test-Path "C:\Program Files (x86)\Inno Setup 6\ISCC.exe") {
    $Iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
}

if ($Iscc) {
    Write-Host "Compiling Inno Setup Installer..." -ForegroundColor Cyan
    & $Iscc (Join-Path $ProjectRoot "installer.iss")
    Write-Host "✓ Installer compiled in: dist\WhisperStudio-Setup-x64.exe" -ForegroundColor Green
} else {
    Write-Host "Notice: Inno Setup compiler (ISCC) not found on this machine." -ForegroundColor Yellow
    Write-Host "GitHub Actions will compile the installer automatically upon push/release." -ForegroundColor Yellow
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "   PACKAGING COMPLETE! Artifacts available in 'dist/'      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

