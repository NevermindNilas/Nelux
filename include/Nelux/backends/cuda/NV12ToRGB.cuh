/*
 * NV12ToRGB.cuh - CUDA YUV to RGB conversion kernels header
 * 
 * Public API for high-performance YUV to RGB color space conversion.
 * 
 * Supports:
 * - NV12 (8-bit 4:2:0), P016 (10-bit 4:2:0)
 * - NV16 (8-bit 4:2:2), P216 (10-bit 4:2:2)
 * - YUV444 (8-bit 4:4:4), YUV444P16 (16-bit 4:4:4)
 * - Multiple color standards: BT.601, BT.709, BT.2020, FCC, SMPTE240M
 * - Limited range (16-235) and full range (0-255) support
 * - Packed RGB24 and planar RGBP output formats
 * 
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifdef NELUX_ENABLE_CUDA

#include <cuda_runtime.h>
#include <cstdint>

namespace nelux::backends::cuda
{

//------------------------------------------------------------------------------
// Color Space Standards
// Values match FFmpeg's AVCOL_SPC_* where applicable
//------------------------------------------------------------------------------
enum ColorSpaceStandard {
    ColorSpaceStandard_BT709 = 1,       // HD content (most common)
    ColorSpaceStandard_Unspecified = 2, // Will default to BT.709
    ColorSpaceStandard_FCC = 4,         // FCC Title 47 (legacy NTSC)
    ColorSpaceStandard_BT470BG = 5,     // BT.470 System B/G (PAL)
    ColorSpaceStandard_BT601 = 6,       // SD content (NTSC/PAL)
    ColorSpaceStandard_SMPTE240M = 7,   // Early HDTV (1988-1998)
    ColorSpaceStandard_BT2020 = 9,      // HDR/UHD content
    ColorSpaceStandard_BT2020C = 10     // BT.2020 constant luminance
};

//------------------------------------------------------------------------------
// Color Range
//------------------------------------------------------------------------------
enum ColorRange {
    ColorRange_Limited = 0,  // Y: 16-235, UV: 16-240 (MPEG/TV range)
    ColorRange_Full = 1      // Y: 0-255, UV: 0-255 (JPEG/PC range)
};

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

/**
 * @brief Initialize color space conversion matrix
 * 
 * @param colorSpace Color space standard (see ColorSpaceStandard enum)
 * @param colorRange Color range (see ColorRange enum)
 * @param stream CUDA stream
 */
void initColorSpaceMatrix(int colorSpace, int colorRange, cudaStream_t stream);
void initColorSpaceMatrix(int colorSpace, cudaStream_t stream);

//------------------------------------------------------------------------------
// NV12 (4:2:0, 8-bit) Functions
//------------------------------------------------------------------------------

/**
 * @brief Convert NV12 to RGB24 (packed)
 * 
 * @param pNv12 NV12 data (Y plane followed by UV plane)
 * @param nNv12Pitch Pitch of NV12 data in bytes
 * @param pRgb RGB24 output buffer (H*W*3 bytes)
 * @param nRgbPitch Pitch of RGB buffer in bytes
 * @param nWidth Frame width in pixels
 * @param nHeight Frame height in pixels
 * @param colorSpace Color space standard
 * @param colorRange Color range
 * @param stream CUDA stream
 */
void launchNv12ToRgb24(
    const uint8_t* pNv12,
    int nNv12Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

// Overload without colorRange (defaults to limited)
void launchNv12ToRgb24(
    const uint8_t* pNv12,
    int nNv12Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    cudaStream_t stream);

/**
 * @brief Convert NV12 to planar RGB (RGBP) for ML workflows
 * Output layout: [R plane (H*W)][G plane (H*W)][B plane (H*W)]
 * Total size: H*W*3 bytes, but organized as CHW instead of HWC
 */
void launchNv12ToRgbPlanar(
    const uint8_t* pNv12,
    int nNv12Pitch,
    uint8_t* pRgbp,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

/**
 * @brief Convert NV12 to RGB24 with separate Y and UV plane pointers
 * Use this when AVFrame provides separate data[0] (Y) and data[1] (UV)
 */
void launchNv12ToRgb24Separate(
    const uint8_t* pY,
    const uint8_t* pUV,
    int nYPitch,
    int nUVPitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// P016 (4:2:0, 10/16-bit) Functions - HDR Content
//------------------------------------------------------------------------------

/**
 * @brief Convert P016 (10-bit NV12) to RGB24
 */
void launchP016ToRgb24(
    const uint8_t* pP016,
    int nP016Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

// Overload without colorRange
void launchP016ToRgb24(
    const uint8_t* pP016,
    int nP016Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    cudaStream_t stream);

/**
 * @brief Convert P016 to planar RGB
 */
void launchP016ToRgbPlanar(
    const uint8_t* pP016,
    int nP016Pitch,
    uint8_t* pRgbp,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// NV16 (4:2:2, 8-bit) Functions - Professional Video
//------------------------------------------------------------------------------

/**
 * @brief Convert NV16 (4:2:2) to RGB24
 */
void launchNv16ToRgb24(
    const uint8_t* pNv16,
    int nNv16Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

/**
 * @brief Convert NV16 to planar RGB
 */
void launchNv16ToRgbPlanar(
    const uint8_t* pNv16,
    int nNv16Pitch,
    uint8_t* pRgbp,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// P216 (4:2:2, 10/16-bit) Functions - Professional HDR Video
//------------------------------------------------------------------------------

/**
 * @brief Convert P216 (10-bit 4:2:2) to RGB24
 */
void launchP216ToRgb24(
    const uint8_t* pP216,
    int nP216Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// YUV444 (4:4:4, 8-bit) Functions - High Quality
//------------------------------------------------------------------------------

/**
 * @brief Convert YUV444 planar to RGB24
 */
void launchYuv444ToRgb24(
    const uint8_t* pY,
    const uint8_t* pU,
    const uint8_t* pV,
    int nYuvPitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

/**
 * @brief Convert YUV444 planar to planar RGB
 */
void launchYuv444ToRgbPlanar(
    const uint8_t* pY,
    const uint8_t* pU,
    const uint8_t* pV,
    int nYuvPitch,
    uint8_t* pRgbp,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// YUV444P16 (4:4:4, 16-bit) Functions - Professional HDR
//------------------------------------------------------------------------------

/**
 * @brief Convert YUV444 16-bit planar to RGB24
 */
void launchYuv444P16ToRgb24(
    const uint8_t* pY,
    const uint8_t* pU,
    const uint8_t* pV,
    int nYuvPitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// Legacy API (Backwards Compatibility)
//------------------------------------------------------------------------------

/**
 * @brief Legacy NV12 to RGB conversion (BT.709, limited range)
 */
void launchNV12ToRGBKernel(
    const uint8_t* yPlane,
    uint8_t* rgbOutput,
    int width,
    int height,
    int yPitch,
    int rgbPitch,
    cudaStream_t stream);

/**
 * @brief Legacy NV12 to RGB with separate UV plane (BT.709, limited range)
 */
void launchNV12ToRGBKernelWithUV(
    const uint8_t* yPlane,
    const uint8_t* uvPlane,
    uint8_t* rgbOutput,
    int width,
    int height,
    int yPitch,
    int uvPitch,
    int rgbPitch,
    cudaStream_t stream);

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
