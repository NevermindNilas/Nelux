# download_ffmpeg.ps1
# Downloads and extracts pre-built FFmpeg shared libraries for NeLux

param(
    [string]$OutputDir = "$PSScriptRoot\..\external\ffmpeg",
    [string]$FFmpegVersion = "7.1.1",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# FFmpeg shared build from gyan.dev (GPL, includes x264/x265)
$FFmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z"
$ArchiveName = "ffmpeg-release-full-shared.7z"

Write-Host "=== FFmpeg Download Script ===" -ForegroundColor Cyan
Write-Host "Output directory: $OutputDir" -ForegroundColor Gray

# Check if already exists
if ((Test-Path "$OutputDir\bin\avcodec-*.dll") -and -not $Force) {
    Write-Host "FFmpeg already exists at $OutputDir. Use -Force to re-download." -ForegroundColor Yellow
    exit 0
}

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
}

$TempDir = Join-Path $env:TEMP "ffmpeg_download"
$ArchivePath = Join-Path $TempDir $ArchiveName

# Create temp directory
if (-not (Test-Path $TempDir)) {
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
}

# Download FFmpeg archive
Write-Host "`n[1/3] Downloading FFmpeg shared build..." -ForegroundColor Green
if (-not (Test-Path $ArchivePath) -or $Force) {
    try {
        Start-BitsTransfer -Source $FFmpegUrl -Destination $ArchivePath
        Write-Host "  Downloaded: $ArchivePath"
    }
    catch {
        Write-Host "  Failed to download from gyan.dev, trying GitHub mirror..." -ForegroundColor Yellow
        $GithubUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip"
        Start-BitsTransfer -Source $GithubUrl -Destination "$TempDir\ffmpeg.zip"
        $ArchivePath = "$TempDir\ffmpeg.zip"
    }
}
else {
    Write-Host "  Using cached: $ArchivePath"
}

# Extract archive
Write-Host "`n[2/3] Extracting FFmpeg..." -ForegroundColor Green
$ExtractDir = Join-Path $TempDir "extracted"
if (Test-Path $ExtractDir) {
    Remove-Item -Recurse -Force $ExtractDir
}
New-Item -ItemType Directory -Path $ExtractDir -Force | Out-Null

if ($ArchivePath -like "*.7z") {
    # Use 7z if available, otherwise try tar (Windows 10+)
    $7zPath = Get-Command "7z" -ErrorAction SilentlyContinue
    if ($7zPath) {
        & 7z x $ArchivePath -o"$ExtractDir" -y | Out-Null
    }
    else {
        # Try with tar (Windows 10 1803+)
        Write-Host "  7z not found, trying PowerShell extraction..." -ForegroundColor Yellow
        # For .7z, we need 7-zip. Download portable version if not found.
        $7zUrl = "https://www.7-zip.org/a/7zr.exe"
        $7zExe = Join-Path $TempDir "7zr.exe"
        if (-not (Test-Path $7zExe)) {
            Start-BitsTransfer -Source $7zUrl -Destination $7zExe
        }
        & $7zExe x $ArchivePath -o"$ExtractDir" -y | Out-Null
    }
}
else {
    Expand-Archive -Path $ArchivePath -DestinationPath $ExtractDir -Force
}

# Find the extracted FFmpeg folder
$FFmpegDir = Get-ChildItem -Path $ExtractDir -Directory | Select-Object -First 1
if (-not $FFmpegDir) {
    throw "Failed to find extracted FFmpeg directory"
}
Write-Host "  Extracted to: $($FFmpegDir.FullName)"

# Copy to output directory
Write-Host "`n[3/3] Installing FFmpeg to $OutputDir..." -ForegroundColor Green

# Clean output directory
if (Test-Path $OutputDir) {
    Remove-Item -Recurse -Force $OutputDir
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# Copy bin, lib, include directories
$Subdirs = @("bin", "lib", "include")
foreach ($subdir in $Subdirs) {
    $SourcePath = Join-Path $FFmpegDir.FullName $subdir
    if (Test-Path $SourcePath) {
        Copy-Item -Path $SourcePath -Destination $OutputDir -Recurse -Force
        $count = (Get-ChildItem -Path (Join-Path $OutputDir $subdir) -File).Count
        Write-Host "  Copied $subdir/ ($count files)"
    }
}

# Verify installation (version-agnostic patterns)
$RequiredDlls = @(
    "avcodec-*.dll",
    "avformat-*.dll", 
    "avutil-*.dll",
    "swscale-*.dll",
    "swresample-*.dll",
    "avfilter-*.dll",
    "avdevice-*.dll"
)

Write-Host "`n=== Verifying Installation ===" -ForegroundColor Cyan
$AllFound = $true
foreach ($pattern in $RequiredDlls) {
    $DllMatches = Get-ChildItem -Path "$OutputDir\bin" -Filter $pattern -ErrorAction SilentlyContinue
    if ($DllMatches) {
        Write-Host "  [OK] $($DllMatches.Name)" -ForegroundColor Green
    }
    else {
        Write-Host "  [MISSING] $pattern" -ForegroundColor Red
        $AllFound = $false
    }
}

if ($AllFound) {
    Write-Host "`nFFmpeg installation complete!" -ForegroundColor Green
    
    # Show version info
    $FFmpegExe = Join-Path $OutputDir "bin\ffmpeg.exe"
    if (Test-Path $FFmpegExe) {
        Write-Host "`nInstalled version:"
        & $FFmpegExe -version | Select-Object -First 1
    }
}
else {
    throw "FFmpeg installation incomplete - some DLLs are missing"
}

# Cleanup temp files
Write-Host "`nCleaning up temporary files..." -ForegroundColor Gray
Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue

Write-Host "`nDone! FFmpeg is ready at: $OutputDir" -ForegroundColor Cyan
