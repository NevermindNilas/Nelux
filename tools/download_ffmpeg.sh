#!/usr/bin/env bash
# download_ffmpeg.sh
# Populate external/ffmpeg/{bin,lib,include} with FFmpeg shared libraries
# and development headers for Nelux builds on Linux / macOS.
#
# Linux: downloads BtbN's GPL-shared prebuilt tarball.
# macOS: requires Homebrew; symlinks `$(brew --prefix ffmpeg)` into external/ffmpeg.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${NELUX_FFMPEG_DIR:-${REPO_ROOT}/external/ffmpeg}"
FORCE="${FORCE:-0}"

say()  { printf '\033[0;36m[ffmpeg]\033[0m %s\n' "$*"; }
ok()   { printf '\033[0;32m  [OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m  [WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[0;31m  [ERR]\033[0m %s\n' "$*" >&2; exit 1; }

os_name="$(uname -s)"
arch_name="$(uname -m)"
say "Target: ${os_name}/${arch_name}  →  ${OUTPUT_DIR}"

if [[ "${FORCE}" != "1" && -f "${OUTPUT_DIR}/include/libavcodec/avcodec.h" ]]; then
  ok "FFmpeg already present at ${OUTPUT_DIR} (set FORCE=1 to re-download)"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"

linux_fetch_btbn() {
  local url tmp extracted_root
  case "${arch_name}" in
    x86_64|amd64)
      url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl-shared.tar.xz"
      ;;
    aarch64|arm64)
      url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-gpl-shared.tar.xz"
      ;;
    *)
      die "Unsupported Linux arch: ${arch_name}"
      ;;
  esac

  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp}"' RETURN

  say "Downloading ${url}"
  curl -fL --retry 3 --retry-delay 2 -o "${tmp}/ffmpeg.tar.xz" "${url}"

  say "Extracting"
  tar -xf "${tmp}/ffmpeg.tar.xz" -C "${tmp}"
  extracted_root="$(find "${tmp}" -maxdepth 1 -mindepth 1 -type d | head -n1)"
  [[ -n "${extracted_root}" ]] || die "archive did not unpack into a directory"

  rm -rf "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"
  cp -R "${extracted_root}/bin"     "${OUTPUT_DIR}/bin"
  cp -R "${extracted_root}/lib"     "${OUTPUT_DIR}/lib"
  cp -R "${extracted_root}/include" "${OUTPUT_DIR}/include"

  # BtbN's GPL-shared tarball ships .so symlinks alongside the versioned SOs.
  # Ensure unversioned symlinks exist — CMake links against libavcodec.so etc.
  (
    cd "${OUTPUT_DIR}/lib"
    for comp in avcodec avformat avutil avfilter swscale avdevice; do
      if [[ ! -e "lib${comp}.so" ]]; then
        target="$(ls -1 lib${comp}.so.* 2>/dev/null | sort -V | tail -n1 || true)"
        [[ -n "${target}" ]] && ln -sf "${target}" "lib${comp}.so" && ok "symlink lib${comp}.so → ${target}"
      fi
    done
  )
}

macos_from_brew() {
  command -v brew >/dev/null 2>&1 || die "Homebrew is required on macOS. Install from https://brew.sh"

  if ! brew list --versions ffmpeg >/dev/null 2>&1; then
    say "Installing ffmpeg via Homebrew"
    brew install ffmpeg
  else
    ok "Homebrew already has ffmpeg ($(brew list --versions ffmpeg))"
  fi

  local prefix
  prefix="$(brew --prefix ffmpeg)"
  [[ -d "${prefix}/include/libavcodec" ]] || die "Homebrew ffmpeg prefix missing headers: ${prefix}"

  rm -rf "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}/bin" "${OUTPUT_DIR}/lib" "${OUTPUT_DIR}/include"

  # Copy so we get a self-contained external/ffmpeg tree and stable paths in
  # CMake-generated build scripts. Use -R to preserve symlinks.
  cp -R "${prefix}/include/"* "${OUTPUT_DIR}/include/"
  cp -R "${prefix}/lib/"libav*.dylib  "${OUTPUT_DIR}/lib/" 2>/dev/null || true
  cp -R "${prefix}/lib/"libsw*.dylib  "${OUTPUT_DIR}/lib/" 2>/dev/null || true
  if [[ -f "${prefix}/bin/ffmpeg" ]]; then
    cp    "${prefix}/bin/ffmpeg"  "${OUTPUT_DIR}/bin/"  || true
    cp    "${prefix}/bin/ffprobe" "${OUTPUT_DIR}/bin/"  || true
  fi
}

case "${os_name}" in
  Linux)  linux_fetch_btbn ;;
  Darwin) macos_from_brew ;;
  *) die "Unsupported OS: ${os_name}" ;;
esac

# Verify result.
required_headers=(libavcodec/avcodec.h libavformat/avformat.h libavutil/avutil.h)
for h in "${required_headers[@]}"; do
  [[ -f "${OUTPUT_DIR}/include/${h}" ]] || die "missing header: ${h}"
done
ok "Headers present under ${OUTPUT_DIR}/include"

if [[ "${os_name}" == "Darwin" ]]; then
  pattern="*.dylib"
else
  pattern="*.so*"
fi
lib_count=$(find "${OUTPUT_DIR}/lib" -maxdepth 1 -name "${pattern}" | wc -l | tr -d ' ')
[[ "${lib_count}" -gt 0 ]] || die "no shared libraries extracted under ${OUTPUT_DIR}/lib"
ok "Found ${lib_count} shared libraries"

say "Done. FFmpeg ready at ${OUTPUT_DIR}"
