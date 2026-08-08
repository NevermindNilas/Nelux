#!/usr/bin/env bash
# download_ffmpeg.sh
#
# Populate external/ffmpeg/{bin,lib,include,licenses} with the FFmpeg build that
# Nelux links against AND ships inside its Linux / macOS wheels.
#
# The binaries come from NevermindNilas/TAS-FFMPEG — our own source build — and
# every URL, hash and version string lives in tools/ffmpeg.lock. Nothing is
# hard-coded here; retarget by editing that file, not this script.
#
# macOS no longer goes through Homebrew. Homebrew is a rolling package manager
# with no pinnable hash, which meant macOS was the one platform that could not
# be held to an exact build. The TAS-FFMPEG macOS tarballs fix that: same
# version, same ABI, same hash discipline as every other platform.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${NELUX_FFMPEG_DIR:-${REPO_ROOT}/external/ffmpeg}"
LOCK_FILE="${NELUX_FFMPEG_LOCK:-${SCRIPT_DIR}/ffmpeg.lock}"
FORCE="${FORCE:-0}"

say()  { printf '\033[0;36m[ffmpeg]\033[0m %s\n' "$*"; }
ok()   { printf '\033[0;32m  [OK]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m  [WARN]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[0;31m  [ERR]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f "${LOCK_FILE}" ]] || die "FFmpeg lock file not found: ${LOCK_FILE}"

lock_value() {
  local key="$1" val
  val="$(sed -n "s/^${key}=//p" "${LOCK_FILE}" | head -n1)"
  [[ -n "${val}" ]] || die "tools/ffmpeg.lock is missing required key '${key}'. A missing platform block is a hard error, never a fall back to the previous release."
  printf '%s' "${val}"
}

# --- Pick the platform block ------------------------------------------------
os_name="$(uname -s)"
arch_name="$(uname -m)"

case "${os_name}/${arch_name}" in
  Linux/x86_64|Linux/amd64)   PLATFORM=LINUX64      ;;
  Linux/aarch64|Linux/arm64)  PLATFORM=LINUXARM64   ;;
  Darwin/arm64)               PLATFORM=MACOS_ARM64  ;;
  Darwin/x86_64)              PLATFORM=MACOS_X64    ;;
  *) die "Unsupported platform: ${os_name}/${arch_name}" ;;
esac

URL="$(lock_value "${PLATFORM}_URL")"
FILE="$(lock_value "${PLATFORM}_FILE")"
ROOT="$(lock_value "${PLATFORM}_ROOT")"
SHA256_WANT="$(lock_value "${PLATFORM}_SHA256")"
VERSION="$(lock_value "${PLATFORM}_VERSION")"
AV_VERSION_INFO="$(lock_value AV_VERSION_INFO)"
STAMP_NAME="$(lock_value PIN_STAMP_FILE)"
SOURCE_URL="$(lock_value CORRESPONDING_SOURCE_URL)"
LICENSE_SPDX="$(lock_value LICENSE_SPDX)"

STAMP_WANTED="${AV_VERSION_INFO} ${SHA256_WANT} ${FILE}"
STAMP_PATH="${OUTPUT_DIR}/${STAMP_NAME}"

say "TAS-FFMPEG ${VERSION} (${AV_VERSION_INFO}) for ${PLATFORM}  →  ${OUTPUT_DIR}"

# --- Stale-tree guard -------------------------------------------------------
if [[ "${FORCE}" != "1" && -f "${STAMP_PATH}" ]]; then
  if [[ "$(cat "${STAMP_PATH}")" == "${STAMP_WANTED}" ]]; then
    ok "FFmpeg ${AV_VERSION_INFO} already installed (pin stamp matches). Set FORCE=1 to re-download."
    exit 0
  fi
fi
if [[ -d "${OUTPUT_DIR}/lib" ]]; then
  warn "Existing FFmpeg tree does not match the pin in tools/ffmpeg.lock — replacing it."
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

# --- Download ---------------------------------------------------------------
say "[1/4] Downloading ${URL}"
curl -fL --retry 3 --retry-delay 2 -o "${tmp}/${FILE}" "${URL}"

# --- Verify -----------------------------------------------------------------
say "[2/4] Verifying SHA256"
if command -v sha256sum >/dev/null 2>&1; then
  sha_actual="$(sha256sum "${tmp}/${FILE}" | awk '{print $1}')"
else
  sha_actual="$(shasum -a 256 "${tmp}/${FILE}" | awk '{print $1}')"
fi
if [[ "${sha_actual}" != "${SHA256_WANT}" ]]; then
  die "SHA256 MISMATCH for ${FILE}
    expected: ${SHA256_WANT}
    actual:   ${sha_actual}
  Refusing to install. Either tools/ffmpeg.lock is stale or the download was
  tampered with. Do NOT 'fix' this by pasting the new hash without checking the
  TAS-FFMPEG release it came from."
fi
ok "SHA256 ${sha_actual}"

# --- Extract ----------------------------------------------------------------
say "[3/4] Extracting"
tar -xf "${tmp}/${FILE}" -C "${tmp}"
src_root="${tmp}/${ROOT}"
[[ -d "${src_root}" ]] || die "Archive did not unpack into the expected root '${ROOT}'. Check ${PLATFORM}_ROOT in tools/ffmpeg.lock."

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# -R preserves the lib*.so -> lib*.so.MAJOR.MINOR.PATCH symlink chain, which
# both the linker (libavcodec.so) and the loader (libavcodec.so.62) need. On
# macOS it preserves the equivalent lib*.dylib chain. licenses/ and
# manifest.json are the GPL paper trail that travels with the binaries into the
# wheel — do not drop them.
for sub in bin lib include licenses; do
  if [[ -d "${src_root}/${sub}" ]]; then
    cp -R "${src_root}/${sub}" "${OUTPUT_DIR}/${sub}"
    ok "${sub}/"
  elif [[ "${sub}" != "licenses" ]]; then
    die "Archive is missing required directory '${sub}'"
  fi
done
for f in manifest.json LICENSE* COPYING* GPL*; do
  [[ -e "${src_root}/${f}" ]] && cp -R "${src_root}/${f}" "${OUTPUT_DIR}/" || true
done

# --- Verify the installed tree ----------------------------------------------
say "[4/4] Verifying installation"

if [[ "${os_name}" == "Darwin" ]]; then
  lib_suffix="dylib"
else
  lib_suffix="so"
fi

for comp in avcodec:AVCODEC avformat:AVFORMAT avutil:AVUTIL avfilter:AVFILTER \
            avdevice:AVDEVICE swscale:SWSCALE swresample:SWRESAMPLE; do
  name="${comp%%:*}"
  major="$(lock_value "SONAME_${comp##*:}")"
  if [[ "${os_name}" == "Darwin" ]]; then
    versioned="${OUTPUT_DIR}/lib/lib${name}.${major}.dylib"
  else
    versioned="${OUTPUT_DIR}/lib/lib${name}.so.${major}"
  fi
  [[ -e "${versioned}" ]] || die "Missing $(basename "${versioned}") — the archive's soname majors disagree with tools/ffmpeg.lock"
  # CMake links against the unversioned name; the archives ship it, but a
  # tar without symlink support (or a copy through a filesystem that drops
  # them) would silently break the link step much later.
  [[ -e "${OUTPUT_DIR}/lib/lib${name}.${lib_suffix}" ]] \
    || ln -sf "$(basename "${versioned}")" "${OUTPUT_DIR}/lib/lib${name}.${lib_suffix}"
done
ok "All seven shared libraries present with the expected soname majors"

for h in libavcodec/avcodec.h libavformat/avformat.h libavutil/avutil.h; do
  [[ -f "${OUTPUT_DIR}/include/${h}" ]] || die "missing header: ${h}"
done
ok "Headers present under ${OUTPUT_DIR}/include"

# The build identity assertion. --extra-version=tas means no distro, gyan or
# BtbN build can produce this string, so this catches "wrong build" as well as
# "wrong version".
if [[ -x "${OUTPUT_DIR}/bin/ffmpeg" ]]; then
  version_line="$("${OUTPUT_DIR}/bin/ffmpeg" -version 2>/dev/null | head -n1 || true)"
  case "${version_line}" in
    *"${AV_VERSION_INFO}"*) ok "${version_line}" ;;
    "") warn "ffmpeg did not run here — skipped the av_version_info assertion" ;;
    *) die "ffmpeg reports '${version_line}' but tools/ffmpeg.lock expects av_version_info '${AV_VERSION_INFO}'" ;;
  esac
else
  warn "bin/ffmpeg absent — skipped the av_version_info assertion"
fi

printf '%s' "${STAMP_WANTED}" > "${STAMP_PATH}"

say "Done. FFmpeg ${AV_VERSION_INFO} ready at ${OUTPUT_DIR}"
say "These binaries are ${LICENSE_SPDX} and are shipped inside the wheel."
say "Corresponding source: ${SOURCE_URL}"
