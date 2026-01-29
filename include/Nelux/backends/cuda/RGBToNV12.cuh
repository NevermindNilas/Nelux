/*
 * RGBToNV12.cuh - Header for CUDA RGB to YUV conversion functions
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
// Color space standards
//------------------------------------------------------------------------------
enum ColorSpaceStandardEncode {
    ColorSpaceEncode_BT709 = 1,
    ColorSpaceEncode_Unspecified = 2,
    ColorSpaceEncode_BT601 = 6,
    ColorSpaceEncode_BT2020 = 9
};

enum ColorRangeEncode {
    ColorRangeEncode_Limited = 0,
    ColorRangeEncode_Full = 1
};

//------------------------------------------------------------------------------
// RGB to YUV conversion functions
//------------------------------------------------------------------------------

/**
 * @brief Set the RGB to YUV conversion matrix
 */
void SetMatRgb2Yuv(int colorSpace, int colorRange, cudaStream_t stream);

/**
 * @brief Convert RGB24 to NV12 (8-bit 4:2:0)
 * 
 * @param pRgb Pointer to RGB24 input buffer (GPU memory)
 * @param nRgbPitch Pitch of RGB buffer in bytes
 * @param pNv12 Pointer to NV12 output buffer (GPU memory)
 * @param nNv12Pitch Pitch of NV12 buffer in bytes
 * @param nWidth Width in pixels
 * @param nHeight Height in pixels
 * @param nSurfaceHeight Height of Y plane in output (for UV offset calculation)
 * @param colorSpace Color space standard (BT.601, BT.709, BT.2020)
 * @param colorRange Color range (Limited or Full)
 * @param stream CUDA stream for async execution
 */
void Rgb24ToNv12(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pNv12,
    int nNv12Pitch,
    int nWidth,
    int nHeight,
    int nSurfaceHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

/**
 * @brief Convert RGB24 to P010 (10-bit 4:2:0)
 */
void Rgb24ToP010(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pP010,
    int nP010Pitch,
    int nWidth,
    int nHeight,
    int nSurfaceHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

/**
 * @brief Convert RGB24 to YUV444P (8-bit 4:4:4 planar)
 */
void Rgb24ToYuv444(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pY,
    uint8_t* pU,
    uint8_t* pV,
    int nYuvPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

/**
 * @brief Convert RGB24 to NV16 (8-bit 4:2:2)
 */
void Rgb24ToNv16(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pNv16,
    int nNv16Pitch,
    int nWidth,
    int nHeight,
    int nSurfaceHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
