/*
 * RGBToYUVExact.cuh - CUDA RGB -> YUV kernels that are byte-exact with
 * nelux's CPU path (RGBToAutoConverter: legacy swscale, SWS_BILINEAR, same
 * size, libswscale 10.1.102 on x86-64).
 *
 * A CUDA tensor converted here and a CPU tensor converted by swscale give
 * the same YUV bytes, so they give the same bitstream: NVENC fed either way
 * matches, and a CUDA tensor can be converted on the GPU before a software
 * encoder (copying only the smaller YUV back to the host).
 *
 * Plans (coefficients, range constants, chroma filter tables) are built on
 * the host by buildRgbToYuvExactPlan (RGBToYUVExactGPU.hpp). Every launch
 * takes its constants as kernel arguments; there is no global state.
 */

#pragma once

#ifdef NELUX_ENABLE_CUDA

#include <cuda_runtime.h>
#include <cstdint>

namespace nelux::backends::cuda
{

// Largest chroma filters the kernels take. swscale's 2:1 bilinear vertical
// filter has 4 taps (5 before alignment for the shortest heights); a plan
// that needs more is rejected and the caller keeps the CPU path.
constexpr int kExactMaxVTaps = 8;
constexpr int kExactMaxHTaps = 8;

// Kernel-side description of one conversion, passed BY VALUE to every
// launch, so two encoders with different matrices on one device can never
// see each other's constants.
struct RgbToYuvExactParams
{
    // RGB2YUV_SHIFT=15 table, swscale's input_rgb2yuv_table entries.
    int32_t ry, gy, by;
    int32_t ru, gu, bu;
    int32_t rv, gv, bv;

    // Full-range pass (lumRangeToJpeg / chrRangeToJpeg on the 15-bit
    // intermediates). Zero for limited range.
    int32_t rangeConvert;
    int32_t lumRangeCoeff, lumRangeOffset;
    int32_t chrRangeCoeff, chrRangeOffset;

    // Vertical chroma filter, 4:2:0 only: vChrTaps taps per chroma row,
    // starting at source row vChrPos[row]. Device pointers.
    const int32_t* vChrPos;
    const int16_t* vChrCoef;
    int32_t vChrTaps;

    // Horizontal chroma filter, odd-width 4:2:x only. Device pointers.
    const int32_t* hChrPos;
    const int16_t* hChrCoef;
    int32_t hChrTaps;
};

enum class ExactRgbSource
{
    Rgb24,  // uint8 R,G,B
    Rgba,   // uint8 R,G,B,A (alpha ignored, as swscale drops it)
    Rgb48,  // uint16 R,G,B
    Rgba64, // uint16 R,G,B,A (alpha ignored)
};

enum class ExactYuvTarget
{
    Yuv420p,   // also yuvj420p (full range)
    Nv12,
    Yuv422p,   // also yuvj422p
    Yuv444p,   // also yuvj444p
    Yuv420p10, // little-endian
    Yuv422p10,
    Yuv444p10,
    P010,      // little-endian, MSB-aligned
};

// What the source elements are. Integer: the source format's own samples
// (uint8 for Rgb24/Rgba, uint16 for Rgb48/Rgba64). The float kinds are a
// [0,1] frame narrowed per sample to exactly what encode_frame's torch
// normalisation gives: (x.float() * 255).clamp(0, 255) truncated to uint8
// for an 8-bit source, (x.float() * 65535).round().clamp(0, 65535) to uint16
// for a 16-bit one, NaN to 0. A float source is read for an 8-bit source
// into an 8-bit target or a 16-bit source into a 10-bit one, the pairs the
// encoder stages a float frame as.
enum class ExactElement
{
    Integer,
    Float32,
    Float16,
    BFloat16,
};

// Source pixels in device memory, in ELEMENTS (bytes for uint8 sources,
// uint16 for 16-bit ones, the float's size for a float source). HWC:
// pixelStride = 3 or 4, channelStride = 1, any rowStride. CHW (planar R, G,
// B): pixelStride = 1, channelStride = H*W (or the tensor's channel
// stride). Any other integer layout should be made contiguous; a float
// source is read through whatever strides it has.
struct RgbToYuvExactSource
{
    const void* data;
    int64_t rowStride;
    int64_t pixelStride;
    int64_t channelStride;
    ExactElement element = ExactElement::Integer;
};

// Destination planes in device memory. plane[1] is the interleaved UV
// plane for NV12/P010 (plane[2] unused). Pitches in bytes.
struct RgbToYuvExactDest
{
    void* plane[3];
    int64_t pitch[3];
};

/**
 * @brief Enqueue the conversion on @p stream. Asynchronous: no allocation,
 *        no synchronisation; the caller orders consumers with an event or a
 *        stream sync. Returns the launch status (cudaGetLastError).
 *
 * @p params must come from buildRgbToYuvExactPlan for the same width,
 * height, source and target, with its filter tables resident on the device
 * (RgbToYuvExactConverter does both).
 */
cudaError_t RgbToYuvExact(const RgbToYuvExactParams& params, ExactRgbSource source,
                          ExactYuvTarget target, int width, int height,
                          const RgbToYuvExactSource& src, const RgbToYuvExactDest& dst,
                          cudaStream_t stream);

/**
 * @brief The float narrowing alone: @p count float elements of kind
 *        @p element at @p src become the uint8 (@p sampleBits 8) or uint16
 *        (16) samples the kernels read, written as uint16. The same device
 *        code as the converts; for checking it against torch.
 */
cudaError_t RgbToYuvExactNarrow(ExactElement element, int sampleBits, const void* src,
                                int64_t count, uint16_t* dst, cudaStream_t stream);

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
