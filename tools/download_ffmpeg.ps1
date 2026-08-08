# download_ffmpeg.ps1
#
# Populate external/ffmpeg/{bin,lib,include,licenses} with the FFmpeg build that
# Nelux links against AND ships inside its Windows wheels.
#
# The binaries come from NevermindNilas/TAS-FFMPEG - our own source build - and
# every URL, hash and version string lives in tools/ffmpeg.lock. Nothing is
# hard-coded here; retarget by editing that file, not this one.
#
# The archive is verified against its SHA256 before a single file is extracted,
# and the installed tree is stamped with what the lock asked for so the
# "already present" short circuit cannot silently keep a stale generation. That
# matters because CMakeLists.txt globs external/ffmpeg/bin/av*.dll at CONFIGURE
# time to bake /DELAYLOAD:<filename> - a stale tree yields a wheel whose
# delay-load contract names the wrong soname.

param(
    [string]$OutputDir = "$PSScriptRoot\..\external\ffmpeg",
    [string]$LockFile = "$PSScriptRoot\ffmpeg.lock",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Say  { param($m) Write-Host "[ffmpeg] $m" -ForegroundColor Cyan }
function Ok   { param($m) Write-Host "  [OK] $m" -ForegroundColor Green }
function Warn { param($m) Write-Host "  [WARN] $m" -ForegroundColor Yellow }

# --- Parse the lock ---------------------------------------------------------
if (-not (Test-Path $LockFile)) {
    throw "FFmpeg lock file not found: $LockFile"
}

$Lock = @{}
foreach ($line in Get-Content $LockFile) {
    if ($line -match '^\s*#') { continue }
    if ($line -match '^([A-Z0-9_]+)=(.*)$') { $Lock[$Matches[1]] = $Matches[2].Trim() }
}

function LockValue {
    param([string]$Key)
    if (-not $Lock.ContainsKey($Key) -or [string]::IsNullOrWhiteSpace($Lock[$Key])) {
        throw "tools/ffmpeg.lock is missing required key '$Key'. A missing platform block is a hard error, never a fall back to the previous release."
    }
    return $Lock[$Key]
}

$Url        = LockValue "WIN64_URL"
$ArchiveName= LockValue "WIN64_FILE"
$Root       = LockValue "WIN64_ROOT"
$Sha256     = (LockValue "WIN64_SHA256").ToLowerInvariant()
$Version    = LockValue "WIN64_VERSION"
$VersionInfo= LockValue "AV_VERSION_INFO"
$StampName  = LockValue "PIN_STAMP_FILE"
$SourceUrl  = LockValue "CORRESPONDING_SOURCE_URL"
$LicenseSpdx= LockValue "LICENSE_SPDX"

# Identity the installed tree must carry to be considered current.
$StampWanted = "$VersionInfo $Sha256 $ArchiveName"
$StampPath   = Join-Path $OutputDir $StampName

Say "TAS-FFMPEG $Version ($VersionInfo) -> $OutputDir"

# --- Stale-tree guard -------------------------------------------------------
if (-not $Force) {
    # A truncated stamp (interrupted Set-Content below) reads back as $null, so
    # test the value before calling Trim() on it - an empty stamp means stale,
    # not a crash.
    $StampFound = if (Test-Path $StampPath) { Get-Content $StampPath -Raw } else { $null }
    if ($StampFound -and $StampFound.Trim() -eq $StampWanted) {
        Ok "FFmpeg $VersionInfo already installed (pin stamp matches). Use -Force to re-download."
        exit 0
    }
    if (Test-Path (Join-Path $OutputDir "bin")) {
        Warn "Existing FFmpeg tree does not match the pin in tools/ffmpeg.lock - replacing it."
    }
}

$TempDir = Join-Path $env:TEMP "nelux_ffmpeg_download"
if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
$ArchivePath = Join-Path $TempDir $ArchiveName

# --- Download ---------------------------------------------------------------
Say "[1/4] Downloading $Url"
& curl.exe -L --fail --retry 3 --retry-delay 2 -o $ArchivePath $Url
if ($LASTEXITCODE -ne 0) { throw "Download failed ($LASTEXITCODE): $Url" }

# --- Verify -----------------------------------------------------------------
Say "[2/4] Verifying SHA256"
$Actual = (Get-FileHash -Path $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($Actual -ne $Sha256) {
    throw @"
SHA256 MISMATCH for $ArchiveName
  expected: $Sha256
  actual:   $Actual
Refusing to install. Either tools/ffmpeg.lock is stale or the download was
tampered with. Do NOT 'fix' this by pasting the new hash without checking the
TAS-FFMPEG release it came from.
"@
}
Ok "SHA256 $Actual"

# --- Extract ----------------------------------------------------------------
Say "[3/4] Extracting"
$ExtractDir = Join-Path $TempDir "extracted"
Expand-Archive -Path $ArchivePath -DestinationPath $ExtractDir -Force

$SrcRoot = Join-Path $ExtractDir $Root
if (-not (Test-Path $SrcRoot)) {
    throw "Archive did not unpack into the expected root '$Root'. Check WIN64_ROOT in tools/ffmpeg.lock."
}

if (Test-Path $OutputDir) { Remove-Item -Recurse -Force $OutputDir }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

# bin/ carries the DLLs AND the MSVC import libs; lib/ carries the .def files
# CMake turns into import libs plus the MinGW .dll.a; licenses/ and
# manifest.json are the GPL paper trail that has to travel with the binaries
# into the wheel.
foreach ($sub in @("bin", "lib", "include", "licenses")) {
    $src = Join-Path $SrcRoot $sub
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $OutputDir -Recurse -Force
        $count = (Get-ChildItem -Path (Join-Path $OutputDir $sub) -Recurse -File).Count
        Ok "$sub/ ($count files)"
    }
    elseif ($sub -ne "licenses") {
        throw "Archive is missing required directory '$sub'"
    }
}
foreach ($glob in @("manifest.json", "LICENSE*", "COPYING*", "GPL*")) {
    Get-ChildItem -Path $SrcRoot -Filter $glob -File -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-Item $_.FullName -Destination $OutputDir -Force }
}

# --- Verify the installed tree ----------------------------------------------
Say "[4/4] Verifying installation"

$RequiredDlls = @(
    "avcodec-$(LockValue 'SONAME_AVCODEC').dll",
    "avformat-$(LockValue 'SONAME_AVFORMAT').dll",
    "avutil-$(LockValue 'SONAME_AVUTIL').dll",
    "avfilter-$(LockValue 'SONAME_AVFILTER').dll",
    "avdevice-$(LockValue 'SONAME_AVDEVICE').dll",
    "swscale-$(LockValue 'SONAME_SWSCALE').dll",
    "swresample-$(LockValue 'SONAME_SWRESAMPLE').dll"
)
foreach ($dll in $RequiredDlls) {
    if (-not (Test-Path (Join-Path $OutputDir "bin\$dll"))) {
        throw "Missing $dll - the archive's soname majors disagree with tools/ffmpeg.lock"
    }
}
Ok "All seven runtime DLLs present with the expected soname majors"

if (-not (Test-Path (Join-Path $OutputDir "include\libavcodec\avcodec.h"))) {
    throw "Missing headers under $OutputDir\include"
}

# The build identity assertion. --extra-version=tas means no distro, gyan or
# BtbN build can produce this string, so this catches "wrong build" as well as
# "wrong version".
$FFmpegExe = Join-Path $OutputDir "bin\ffmpeg.exe"
if (Test-Path $FFmpegExe) {
    # Capture the whole banner before slicing it. Piping a native command
    # straight into Select-Object -First 1 stops the pipeline early, and
    # Windows PowerShell 5.1 then leaves $LASTEXITCODE = -1 behind - which the
    # GitHub Actions powershell wrapper turns into a failed step even though
    # the install succeeded.
    $VersionOut  = & $FFmpegExe -version 2>&1
    $VersionLine = $VersionOut | Select-Object -First 1
    if ($VersionLine -notmatch [regex]::Escape($VersionInfo)) {
        throw "ffmpeg.exe reports '$VersionLine' but tools/ffmpeg.lock expects av_version_info '$VersionInfo'"
    }
    Ok $VersionLine
}
else {
    Warn "ffmpeg.exe absent - skipped the av_version_info assertion"
}

Set-Content -Path $StampPath -Value $StampWanted -NoNewline
Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue

Say "Done. FFmpeg $VersionInfo ready at $OutputDir"
Say "These binaries are $LicenseSpdx and are shipped inside the wheel."
Say "Corresponding source: $SourceUrl"
