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

/**
 * @brief Convert RGB24 to YUV420P (I420, 8-bit 4:2:0 planar)
 * 
 * @param pRgb Pointer to RGB24 input buffer (GPU memory)
 * @param nRgbPitch Pitch of RGB buffer in bytes
 * @param pY Pointer to Y plane output buffer (GPU memory)
 * @param pU Pointer to U plane output buffer (GPU memory)
 * @param pV Pointer to V plane output buffer (GPU memory)
 * @param nYPitch Pitch of Y plane in bytes (U/V have half pitch)
 * @param nWidth Width in pixels
 * @param nHeight Height in pixels
 * @param colorSpace Color space standard (BT.601, BT.709, BT.2020)
 * @param colorRange Color range (Limited or Full)
 * @param stream CUDA stream for async execution
 */
void Rgb24ToYuv420p(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pY,
    uint8_t* pU,
    uint8_t* pV,
    int nYPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

/**
 * @brief Convert RGB24 to YUV420P16 (16-bit 4:2:0 planar)
 * 
 * @param pRgb Pointer to RGB24 input buffer (GPU memory)
 * @param nRgbPitch Pitch of RGB buffer in bytes
 * @param pY Pointer to Y plane output buffer (GPU memory, 16-bit)
 * @param pU Pointer to U plane output buffer (GPU memory, 16-bit)
 * @param pV Pointer to V plane output buffer (GPU memory, 16-bit)
 * @param nYPitch Pitch of Y plane in bytes (U/V have half pitch)
 * @param nWidth Width in pixels
 * @param nHeight Height in pixels
 * @param colorSpace Color space standard (BT.601, BT.709, BT.2020)
 * @param colorRange Color range (Limited or Full)
 * @param stream CUDA stream for async execution
 */
void Rgb24ToYuv420p16(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pY,
    uint8_t* pU,
    uint8_t* pV,
    int nYPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// PACKED RGB/BGR FORMATS (for NVENC RGB input)
//------------------------------------------------------------------------------

/**
 * @brief Convert RGB24 to BGR0/BGRA (packed BGR for NVENC)
 * 
 * Swaps R and B channels: RGB -> BGR
 * @param addAlpha true for BGRA (4 bytes), false for BGR0 (3 bytes)
 */
void Rgb24ToBgr(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pBgr,
    int nBgrPitch,
    int nWidth,
    int nHeight,
    bool addAlpha,
    cudaStream_t stream);

/**
 * @brief Convert RGB24 to RGB0/RGBA (packed RGB for NVENC)
 * 
 * No channel swap, just optional alpha addition
 * @param addAlpha true for RGBA (4 bytes), false for RGB0 (3 bytes)
 */
void Rgb24ToRgbPacked(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pDst,
    int nDstPitch,
    int nWidth,
    int nHeight,
    bool addAlpha,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// PLANAR RGB FORMATS (GBRP for NVENC)
//------------------------------------------------------------------------------

/**
 * @brief Convert RGB24 to GBRP (planar GBR for NVENC)
 * 
 * Output: 3 separate planes (G, B, R order for FFmpeg)
 * @param nPitch Pitch of each plane in bytes
 */
void Rgb24ToGbrp(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pG,
    uint8_t* pB,
    uint8_t* pR,
    int nPitch,
    int nWidth,
    int nHeight,
    cudaStream_t stream);

/**
 * @brief Convert RGB48 to GBRP16LE (planar 16-bit GBR)
 * 
 * Input: Packed RGB48 (6 bytes per pixel, 16-bit per channel)
 * Output: 3 separate 16-bit planes (G, B, R order)
 * @param nPitch Pitch of each plane in bytes
 */
void Rgb48ToGbrp16(
    const uint8_t* pRgb48,
    int nRgbPitch,
    uint8_t* pG,
    uint8_t* pB,
    uint8_t* pR,
    int nPitch,
    int nWidth,
    int nHeight,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// 4:2:2 FORMATS
//------------------------------------------------------------------------------

/**
 * @brief Convert RGB24 to P210 (10-bit 4:2:2 semi-planar)
 */
void Rgb24ToP210(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pP210,
    int nP210Pitch,
    int nWidth,
    int nHeight,
    int nSurfaceHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

/**
 * @brief Convert RGB24 to P216 (16-bit 4:2:2 semi-planar)
 */
void Rgb24ToP216(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pP216,
    int nP216Pitch,
    int nWidth,
    int nHeight,
    int nSurfaceHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream);

//------------------------------------------------------------------------------
// 4:4:4 HIGH BIT DEPTH FORMATS
//------------------------------------------------------------------------------

/**
 * @brief Convert RGB24 to YUV444P10LE (10-bit planar 4:4:4)
 */
void Rgb24ToYuv444P10(
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
 * @brief Convert RGB24 to YUV444P16LE (16-bit planar 4:4:4)
 */
void Rgb24ToYuv444P16(
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

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
