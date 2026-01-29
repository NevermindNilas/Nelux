/*
 * NV12ToRGB.cu - CUDA kernels for YUV to RGB conversion
 * 
 * Inspired by NVIDIA Video Codec SDK (MIT License)
 * This implementation uses constant memory matrices and vectorized access
 * for high-performance YUV to RGB color space conversion.
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

#ifdef NELUX_ENABLE_CUDA

#include <cuda_runtime.h>
#include <cstdint>

namespace nelux::backends::cuda
{

//------------------------------------------------------------------------------
// Color space standards (matches FFmpeg AVCOL_SPC_* values where applicable)
//------------------------------------------------------------------------------
enum ColorSpaceStandard {
    ColorSpaceStandard_BT709 = 1,       // HD content (most common)
    ColorSpaceStandard_Unspecified = 2, // Will default to BT.709
    ColorSpaceStandard_FCC = 4,         // FCC Title 47
    ColorSpaceStandard_BT470BG = 5,     // BT.470 System B/G (PAL)
    ColorSpaceStandard_BT601 = 6,       // SD content (NTSC/PAL)
    ColorSpaceStandard_SMPTE240M = 7,   // Early HDTV
    ColorSpaceStandard_BT2020 = 9,      // HDR/UHD content
    ColorSpaceStandard_BT2020C = 10     // BT.2020 constant luminance
};

//------------------------------------------------------------------------------
// Color range
//------------------------------------------------------------------------------
enum ColorRange {
    ColorRange_Limited = 0,  // Y: 16-235, UV: 16-240 (MPEG/TV range)
    ColorRange_Full = 1      // Y: 0-255, UV: 0-255 (JPEG/PC range)
};

//------------------------------------------------------------------------------
// Constant memory for YUV to RGB conversion matrix
// Using constant memory for faster access (cached and broadcast to all threads)
//------------------------------------------------------------------------------
__constant__ float matYuv2Rgb[3][3];

//------------------------------------------------------------------------------
// RGB pixel types for vectorized access
//------------------------------------------------------------------------------
union RGB24 {
    uchar3 v;
    struct {
        uint8_t r, g, b;
    } c;
    
    __device__ __host__ RGB24() : v{0, 0, 0} {}
    __device__ __host__ RGB24(uint8_t r_, uint8_t g_, uint8_t b_) { c.r = r_; c.g = g_; c.b = b_; }
};

// For writing two RGB24 pixels at once (6 bytes, vectorized)
struct RGB24x2 {
    uchar3 x;
    uchar3 y;
};

// 32-bit RGBA with alpha
union RGBA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t r, g, b, a;
    } c;
};

// For writing two RGBA32 pixels at once
struct RGBA32x2 {
    uint32_t x;
    uint32_t y;
};

//------------------------------------------------------------------------------
// Helper function to compute the YUV->RGB matrix coefficients
// Based on NVIDIA Video Codec SDK approach with extended color space support
//------------------------------------------------------------------------------
inline void GetColorSpaceConstants(int iMatrix, int colorRange, float &wr, float &wb, 
                                   int &black, int &white, int &max, int &uvMax) {
    // Set range-dependent values
    if (colorRange == ColorRange_Full) {
        black = 0;
        white = 255;
        uvMax = 255;
    } else {
        black = 16;
        white = 235;
        uvMax = 240;
    }
    max = 255;

    switch (iMatrix)
    {
    case ColorSpaceStandard_BT709:
    case ColorSpaceStandard_Unspecified:
    default:
        // BT.709 - HD content (most modern video)
        wr = 0.2126f; wb = 0.0722f;
        break;

    case ColorSpaceStandard_FCC:
        // FCC Title 47 (legacy NTSC)
        wr = 0.30f; wb = 0.11f;
        break;

    case ColorSpaceStandard_BT470BG:
    case ColorSpaceStandard_BT601:
        // BT.601 / BT.470 - SD content
        wr = 0.2990f; wb = 0.1140f;
        break;

    case ColorSpaceStandard_SMPTE240M:
        // SMPTE 240M - Early HDTV (1988-1998)
        wr = 0.212f; wb = 0.087f;
        break;

    case ColorSpaceStandard_BT2020:
    case ColorSpaceStandard_BT2020C:
        // BT.2020 - HDR/UHD content
        wr = 0.2627f; wb = 0.0593f;
        // For 10-bit content with limited range
        if (colorRange == ColorRange_Limited) {
            black = 64;   // 16 << 2 for 10-bit in 8-bit space
            white = 940 >> 2;  // Scaled for 8-bit processing
        }
        break;
    }
}

//------------------------------------------------------------------------------
// Pre-computed YUV to RGB conversion matrices
// These constants match libyuv's validated implementations for color accuracy.
// Each matrix converts (Y-offset, U-128, V-128) directly to RGB (0-255).
//
// For limited range: Y is in [16,235], UV is in [16,240]
// For full range: Y is in [0,255], UV is in [0,255]
//
// Matrix format: mat[row][col] where row is R/G/B and col is Y/U/V coefficient
//------------------------------------------------------------------------------

// BT.601 (SD content) - Limited Range
// Y: 16-235 (219 levels), UV: 16-240 (224 levels, centered at 128)
static const float kMatBT601Limited[3][3] = {
    {1.164384f,  0.000000f,  1.596027f},   // R = 1.164*(Y-16) + 1.596*(V-128)
    {1.164384f, -0.391762f, -0.812968f},   // G = 1.164*(Y-16) - 0.392*(U-128) - 0.813*(V-128)
    {1.164384f,  2.017232f,  0.000000f}    // B = 1.164*(Y-16) + 2.017*(U-128)
};

// BT.601 (SD content) - Full Range
static const float kMatBT601Full[3][3] = {
    {1.000000f,  0.000000f,  1.402000f},   // R = Y + 1.402*(V-128)
    {1.000000f, -0.344136f, -0.714136f},   // G = Y - 0.344*(U-128) - 0.714*(V-128)
    {1.000000f,  1.772000f,  0.000000f}    // B = Y + 1.772*(U-128)
};

// BT.709 (HD content) - Limited Range
// Most common for modern HD video
static const float kMatBT709Limited[3][3] = {
    {1.164384f,  0.000000f,  1.792741f},   // R = 1.164*(Y-16) + 1.793*(V-128)
    {1.164384f, -0.213249f, -0.532909f},   // G = 1.164*(Y-16) - 0.213*(U-128) - 0.533*(V-128)
    {1.164384f,  2.112402f,  0.000000f}    // B = 1.164*(Y-16) + 2.112*(U-128)
};

// BT.709 (HD content) - Full Range
static const float kMatBT709Full[3][3] = {
    {1.000000f,  0.000000f,  1.574800f},   // R = Y + 1.575*(V-128)
    {1.000000f, -0.187324f, -0.468124f},   // G = Y - 0.187*(U-128) - 0.468*(V-128)
    {1.000000f,  1.855600f,  0.000000f}    // B = Y + 1.856*(U-128)
};

// BT.2020 (UHD/HDR content) - Limited Range
static const float kMatBT2020Limited[3][3] = {
    {1.164384f,  0.000000f,  1.678674f},   // R
    {1.164384f, -0.187326f, -0.650424f},   // G
    {1.164384f,  2.141772f,  0.000000f}    // B
};

// BT.2020 (UHD/HDR content) - Full Range
static const float kMatBT2020Full[3][3] = {
    {1.000000f,  0.000000f,  1.474600f},   // R
    {1.000000f, -0.164553f, -0.571353f},   // G
    {1.000000f,  1.881400f,  0.000000f}    // B
};

// SMPTE 240M (Early HDTV) - Limited Range
static const float kMatSMPTE240MLimit[3][3] = {
    {1.164384f,  0.000000f,  1.794107f},   // R
    {1.164384f, -0.257985f, -0.542583f},   // G
    {1.164384f,  2.078705f,  0.000000f}    // B
};

// FCC (Legacy NTSC) - Limited Range
static const float kMatFCCLimited[3][3] = {
    {1.164384f,  0.000000f,  1.639589f},   // R
    {1.164384f, -0.338572f, -0.743210f},   // G
    {1.164384f,  2.032397f,  0.000000f}    // B
};

//------------------------------------------------------------------------------
// Set the YUV to RGB conversion matrix in constant memory
// Uses pre-computed, validated matrices for color accuracy
//------------------------------------------------------------------------------
void SetMatYuv2Rgb(int iMatrix, int colorRange, cudaStream_t stream) {
    const float (*mat)[3] = nullptr;
    
    // Select the appropriate pre-computed matrix based on color space and range
    bool isFullRange = (colorRange == ColorRange_Full);
    
    switch (iMatrix) {
        case ColorSpaceStandard_BT709:
        case ColorSpaceStandard_Unspecified:  // Default to BT.709 for HD
        default:
            mat = isFullRange ? kMatBT709Full : kMatBT709Limited;
            break;
            
        case ColorSpaceStandard_BT601:
        case ColorSpaceStandard_BT470BG:
            mat = isFullRange ? kMatBT601Full : kMatBT601Limited;
            break;
            
        case ColorSpaceStandard_BT2020:
        case ColorSpaceStandard_BT2020C:
            mat = isFullRange ? kMatBT2020Full : kMatBT2020Limited;
            break;
            
        case ColorSpaceStandard_SMPTE240M:
            // SMPTE 240M only has limited range in practice
            mat = kMatSMPTE240MLimit;
            break;
            
        case ColorSpaceStandard_FCC:
            // FCC only has limited range
            mat = kMatFCCLimited;
            break;
    }
    
    cudaMemcpyToSymbolAsync(matYuv2Rgb, mat, sizeof(float) * 9, 0, cudaMemcpyHostToDevice, stream);
}

// Overload for backwards compatibility (defaults to limited range)
void SetMatYuv2Rgb(int iMatrix, cudaStream_t stream) {
    SetMatYuv2Rgb(iMatrix, ColorRange_Limited, stream);
}

//------------------------------------------------------------------------------
// Device helper functions
//------------------------------------------------------------------------------
template<class T>
__device__ __forceinline__ T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

/**
 * @brief Convert a single YUV pixel to RGB using the constant memory matrix
 * 
 * The pre-computed matrices already include:
 * - Y scale factor (1.164 for limited range, 1.0 for full range)
 * - UV scale factors for proper chroma expansion
 * 
 * This kernel handles:
 * - Subtracting Y offset (16 for limited range, 0 for full range)
 * - Subtracting UV offset (128 for 8-bit, scaled for higher bit depths)
 * - Applying the matrix multiplication
 * - Clamping to valid RGB range with rounding
 * 
 * @tparam YuvUnit Type of YUV component (uint8_t for 8-bit, uint16_t for 10/16-bit)
 * @param y Y component
 * @param u U component  
 * @param v V component
 * @param fullRange Whether input is full range (0-255) or limited (16-235)
 * @return RGB24 pixel
 */
template<class YuvUnit>
__device__ __forceinline__ RGB24 YuvToRgbForPixel(YuvUnit y, YuvUnit u, YuvUnit v, bool fullRange = false) {
    const int bitDepth = sizeof(YuvUnit) * 8;
    
    // For 8-bit: low=16 (limited) or 0 (full), mid=128
    // For 10-bit: low=64 (limited) or 0 (full), mid=512
    // For 16-bit: low=4096 (limited) or 0 (full), mid=32768
    const int low = fullRange ? 0 : (1 << (bitDepth - 4));   // Y offset
    const int mid = 1 << (bitDepth - 1);                      // UV offset (128 for 8-bit)
    
    // Normalize to 8-bit equivalent values before matrix multiplication
    // This ensures our 8-bit calibrated matrices work for all bit depths
    float normScale = (bitDepth > 8) ? (255.0f / static_cast<float>((1 << bitDepth) - 1)) : 1.0f;
    float lowNorm = static_cast<float>(low) * normScale;
    float midNorm = static_cast<float>(mid) * normScale;
    
    // Convert to normalized values (0-255 equivalent scale)
    float fy = static_cast<float>(y) * normScale - lowNorm;
    float fu = static_cast<float>(u) * normScale - midNorm;
    float fv = static_cast<float>(v) * normScale - midNorm;
    
    // Apply YUV to RGB matrix multiplication
    // The matrix coefficients are calibrated for 8-bit equivalent inputs
    float rf = matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv;
    float gf = matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv;
    float bf = matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv;
    
    // Round and clamp to [0, 255]
    RGB24 rgb;
    rgb.c.r = static_cast<uint8_t>(Clamp(rf + 0.5f, 0.0f, 255.0f));
    rgb.c.g = static_cast<uint8_t>(Clamp(gf + 0.5f, 0.0f, 255.0f));
    rgb.c.b = static_cast<uint8_t>(Clamp(bf + 0.5f, 0.0f, 255.0f));
    return rgb;
}

/**
 * @brief Convert YUV to RGBA32 with alpha channel
 */
template<class YuvUnit>
__device__ __forceinline__ RGBA32 YuvToRgbaForPixel(YuvUnit y, YuvUnit u, YuvUnit v, uint8_t alpha = 255, bool fullRange = false) {
    RGB24 rgb = YuvToRgbForPixel(y, u, v, fullRange);
    RGBA32 rgba;
    rgba.c.r = rgb.c.r;
    rgba.c.g = rgb.c.g;
    rgba.c.b = rgb.c.b;
    rgba.c.a = alpha;
    return rgba;
}

//==============================================================================
// NV12 KERNELS (4:2:0, 8-bit)
//==============================================================================

/**
 * @brief NV12 to RGB24 kernel - processes 2x2 pixel blocks with vectorized writes
 */
__global__ void Nv12ToRgb24Kernel(
    const uint8_t* __restrict__ pNv12,
    int nNv12Pitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    // Each thread processes a 2x2 block of pixels
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Vectorized Y reads (2 pixels at once)
    const uint8_t* pSrcY = pNv12 + y * nNv12Pitch + x;
    uchar2 y0 = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 y1 = *reinterpret_cast<const uchar2*>(pSrcY + nNv12Pitch);
    
    // Read UV pair (shared by 2x2 block)
    const uint8_t* pSrcUV = pNv12 + nSurfaceHeight * nNv12Pitch + (y / 2) * nNv12Pitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert 4 pixels
    RGB24 rgb00 = YuvToRgbForPixel<uint8_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint8_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint8_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint8_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Vectorized RGB writes - write 2 pixels per row at once
    RGB24x2* pDst0 = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    RGB24x2* pDst1 = reinterpret_cast<RGB24x2*>(pRgb + (y + 1) * nRgbPitch + x * 3);
    
    *pDst0 = RGB24x2{rgb00.v, rgb01.v};
    *pDst1 = RGB24x2{rgb10.v, rgb11.v};
}

/**
 * @brief NV12 to planar RGB (RGBP) kernel for ML workflows
 * Output: R plane [0:H*W], G plane [H*W:2*H*W], B plane [2*H*W:3*H*W]
 */
__global__ void Nv12ToRgbPlanarKernel(
    const uint8_t* __restrict__ pNv12,
    int nNv12Pitch,
    uint8_t* __restrict__ pRgbp,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    const int planeSize = nWidth * nHeight;
    
    // Read Y values
    const uint8_t* pSrcY = pNv12 + y * nNv12Pitch + x;
    uchar2 y0 = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 y1 = *reinterpret_cast<const uchar2*>(pSrcY + nNv12Pitch);
    
    // Read UV
    const uint8_t* pSrcUV = pNv12 + nSurfaceHeight * nNv12Pitch + (y / 2) * nNv12Pitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert
    RGB24 rgb00 = YuvToRgbForPixel<uint8_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint8_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint8_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint8_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Write to planar format - R plane
    int idx0 = y * nWidth + x;
    int idx1 = (y + 1) * nWidth + x;
    
    // Vectorized writes to each plane
    *reinterpret_cast<uchar2*>(pRgbp + idx0) = make_uchar2(rgb00.c.r, rgb01.c.r);
    *reinterpret_cast<uchar2*>(pRgbp + idx1) = make_uchar2(rgb10.c.r, rgb11.c.r);
    
    // G plane
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx0) = make_uchar2(rgb00.c.g, rgb01.c.g);
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx1) = make_uchar2(rgb10.c.g, rgb11.c.g);
    
    // B plane
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx0) = make_uchar2(rgb00.c.b, rgb01.c.b);
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx1) = make_uchar2(rgb10.c.b, rgb11.c.b);
}

/**
 * @brief NV12 to RGB24 with separate Y and UV plane pointers
 */
__global__ void Nv12SeparateToRgb24Kernel(
    const uint8_t* __restrict__ pY,
    const uint8_t* __restrict__ pUV,
    int nYPitch,
    int nUVPitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Read Y
    const uint8_t* pSrcY = pY + y * nYPitch + x;
    uchar2 y0 = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 y1 = *reinterpret_cast<const uchar2*>(pSrcY + nYPitch);
    
    // Read UV
    const uint8_t* pSrcUV = pUV + (y / 2) * nUVPitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert
    RGB24 rgb00 = YuvToRgbForPixel<uint8_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint8_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint8_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint8_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Vectorized writes
    RGB24x2* pDst0 = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    RGB24x2* pDst1 = reinterpret_cast<RGB24x2*>(pRgb + (y + 1) * nRgbPitch + x * 3);
    
    *pDst0 = RGB24x2{rgb00.v, rgb01.v};
    *pDst1 = RGB24x2{rgb10.v, rgb11.v};
}

//==============================================================================
// P016 KERNELS (4:2:0, 10/16-bit)
//==============================================================================

/**
 * @brief P016 (10-bit NV12) to RGB24 kernel for HDR content
 */
__global__ void P016ToRgb24Kernel(
    const uint8_t* __restrict__ pP016,
    int nP016Pitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Read 16-bit Y values
    const uint16_t* pSrcY0 = reinterpret_cast<const uint16_t*>(pP016 + y * nP016Pitch) + x;
    const uint16_t* pSrcY1 = reinterpret_cast<const uint16_t*>(pP016 + (y + 1) * nP016Pitch) + x;
    ushort2 y0 = *reinterpret_cast<const ushort2*>(pSrcY0);
    ushort2 y1 = *reinterpret_cast<const ushort2*>(pSrcY1);
    
    // Read 16-bit UV
    const uint16_t* pSrcUV = reinterpret_cast<const uint16_t*>(pP016 + nSurfaceHeight * nP016Pitch + (y / 2) * nP016Pitch) + x;
    ushort2 uv = *reinterpret_cast<const ushort2*>(pSrcUV);
    
    // Convert
    RGB24 rgb00 = YuvToRgbForPixel<uint16_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint16_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint16_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint16_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Vectorized writes
    RGB24x2* pDst0 = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    RGB24x2* pDst1 = reinterpret_cast<RGB24x2*>(pRgb + (y + 1) * nRgbPitch + x * 3);
    
    *pDst0 = RGB24x2{rgb00.v, rgb01.v};
    *pDst1 = RGB24x2{rgb10.v, rgb11.v};
}

/**
 * @brief P016 to planar RGB for ML workflows
 */
__global__ void P016ToRgbPlanarKernel(
    const uint8_t* __restrict__ pP016,
    int nP016Pitch,
    uint8_t* __restrict__ pRgbp,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    const int planeSize = nWidth * nHeight;
    
    // Read 16-bit Y values
    const uint16_t* pSrcY0 = reinterpret_cast<const uint16_t*>(pP016 + y * nP016Pitch) + x;
    const uint16_t* pSrcY1 = reinterpret_cast<const uint16_t*>(pP016 + (y + 1) * nP016Pitch) + x;
    ushort2 y0 = *reinterpret_cast<const ushort2*>(pSrcY0);
    ushort2 y1 = *reinterpret_cast<const ushort2*>(pSrcY1);
    
    // Read 16-bit UV
    const uint16_t* pSrcUV = reinterpret_cast<const uint16_t*>(pP016 + nSurfaceHeight * nP016Pitch + (y / 2) * nP016Pitch) + x;
    ushort2 uv = *reinterpret_cast<const ushort2*>(pSrcUV);
    
    // Convert
    RGB24 rgb00 = YuvToRgbForPixel<uint16_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint16_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint16_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint16_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Write to planar format
    int idx0 = y * nWidth + x;
    int idx1 = (y + 1) * nWidth + x;
    
    *reinterpret_cast<uchar2*>(pRgbp + idx0) = make_uchar2(rgb00.c.r, rgb01.c.r);
    *reinterpret_cast<uchar2*>(pRgbp + idx1) = make_uchar2(rgb10.c.r, rgb11.c.r);
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx0) = make_uchar2(rgb00.c.g, rgb01.c.g);
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx1) = make_uchar2(rgb10.c.g, rgb11.c.g);
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx0) = make_uchar2(rgb00.c.b, rgb01.c.b);
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx1) = make_uchar2(rgb10.c.b, rgb11.c.b);
}

//==============================================================================
// NV16 KERNELS (4:2:2, 8-bit) - Professional video
//==============================================================================

/**
 * @brief NV16 (4:2:2) to RGB24 kernel
 * UV has same height as Y but half width
 */
__global__ void Nv16ToRgb24Kernel(
    const uint8_t* __restrict__ pNv16,
    int nNv16Pitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    // Each thread processes 2 horizontal pixels (they share UV)
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read 2 Y values
    const uint8_t* pSrcY = pNv16 + y * nNv16Pitch + x;
    uchar2 yy = *reinterpret_cast<const uchar2*>(pSrcY);
    
    // Read UV (same row, half width) - UV plane starts at nSurfaceHeight
    const uint8_t* pSrcUV = pNv16 + nSurfaceHeight * nNv16Pitch + y * nNv16Pitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert 2 pixels
    RGB24 rgb0 = YuvToRgbForPixel<uint8_t>(yy.x, uv.x, uv.y, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint8_t>(yy.y, uv.x, uv.y, fullRange);
    
    // Vectorized write
    RGB24x2* pDst = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    *pDst = RGB24x2{rgb0.v, rgb1.v};
}

/**
 * @brief NV16 to planar RGB
 */
__global__ void Nv16ToRgbPlanarKernel(
    const uint8_t* __restrict__ pNv16,
    int nNv16Pitch,
    uint8_t* __restrict__ pRgbp,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    const int planeSize = nWidth * nHeight;
    
    const uint8_t* pSrcY = pNv16 + y * nNv16Pitch + x;
    uchar2 yy = *reinterpret_cast<const uchar2*>(pSrcY);
    
    const uint8_t* pSrcUV = pNv16 + nSurfaceHeight * nNv16Pitch + y * nNv16Pitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    RGB24 rgb0 = YuvToRgbForPixel<uint8_t>(yy.x, uv.x, uv.y, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint8_t>(yy.y, uv.x, uv.y, fullRange);
    
    int idx = y * nWidth + x;
    *reinterpret_cast<uchar2*>(pRgbp + idx) = make_uchar2(rgb0.c.r, rgb1.c.r);
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx) = make_uchar2(rgb0.c.g, rgb1.c.g);
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx) = make_uchar2(rgb0.c.b, rgb1.c.b);
}

//==============================================================================
// P216 KERNELS (4:2:2, 10/16-bit) - Professional HDR video
//==============================================================================

/**
 * @brief P216 (10-bit 4:2:2) to RGB24 kernel
 */
__global__ void P216ToRgb24Kernel(
    const uint8_t* __restrict__ pP216,
    int nP216Pitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nSurfaceHeight,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read 16-bit Y values
    const uint16_t* pSrcY = reinterpret_cast<const uint16_t*>(pP216 + y * nP216Pitch) + x;
    ushort2 yy = *reinterpret_cast<const ushort2*>(pSrcY);
    
    // Read 16-bit UV
    const uint16_t* pSrcUV = reinterpret_cast<const uint16_t*>(pP216 + nSurfaceHeight * nP216Pitch + y * nP216Pitch) + x;
    ushort2 uv = *reinterpret_cast<const ushort2*>(pSrcUV);
    
    // Convert
    RGB24 rgb0 = YuvToRgbForPixel<uint16_t>(yy.x, uv.x, uv.y, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint16_t>(yy.y, uv.x, uv.y, fullRange);
    
    // Vectorized write
    RGB24x2* pDst = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    *pDst = RGB24x2{rgb0.v, rgb1.v};
}

//==============================================================================
// YUV444 KERNELS (4:4:4, 8-bit) - High quality, no chroma subsampling
//==============================================================================

/**
 * @brief YUV444 planar to RGB24 kernel
 * Each pixel has its own U and V values
 */
__global__ void Yuv444ToRgb24Kernel(
    const uint8_t* __restrict__ pY,
    const uint8_t* __restrict__ pU,
    const uint8_t* __restrict__ pV,
    int nYuvPitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    bool fullRange)
{
    // Each thread processes 2 horizontal pixels
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read Y, U, V (each pixel has its own values)
    const uint8_t* pSrcY = pY + y * nYuvPitch + x;
    const uint8_t* pSrcU = pU + y * nYuvPitch + x;
    const uint8_t* pSrcV = pV + y * nYuvPitch + x;
    
    uchar2 yy = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 uu = *reinterpret_cast<const uchar2*>(pSrcU);
    uchar2 vv = *reinterpret_cast<const uchar2*>(pSrcV);
    
    // Convert (each pixel gets its own U, V)
    RGB24 rgb0 = YuvToRgbForPixel<uint8_t>(yy.x, uu.x, vv.x, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint8_t>(yy.y, uu.y, vv.y, fullRange);
    
    // Vectorized write
    RGB24x2* pDst = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    *pDst = RGB24x2{rgb0.v, rgb1.v};
}

/**
 * @brief YUV444 planar to planar RGB
 */
__global__ void Yuv444ToRgbPlanarKernel(
    const uint8_t* __restrict__ pY,
    const uint8_t* __restrict__ pU,
    const uint8_t* __restrict__ pV,
    int nYuvPitch,
    uint8_t* __restrict__ pRgbp,
    int nWidth,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    const int planeSize = nWidth * nHeight;
    
    const uint8_t* pSrcY = pY + y * nYuvPitch + x;
    const uint8_t* pSrcU = pU + y * nYuvPitch + x;
    const uint8_t* pSrcV = pV + y * nYuvPitch + x;
    
    uchar2 yy = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 uu = *reinterpret_cast<const uchar2*>(pSrcU);
    uchar2 vv = *reinterpret_cast<const uchar2*>(pSrcV);
    
    RGB24 rgb0 = YuvToRgbForPixel<uint8_t>(yy.x, uu.x, vv.x, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint8_t>(yy.y, uu.y, vv.y, fullRange);
    
    int idx = y * nWidth + x;
    *reinterpret_cast<uchar2*>(pRgbp + idx) = make_uchar2(rgb0.c.r, rgb1.c.r);
    *reinterpret_cast<uchar2*>(pRgbp + planeSize + idx) = make_uchar2(rgb0.c.g, rgb1.c.g);
    *reinterpret_cast<uchar2*>(pRgbp + 2 * planeSize + idx) = make_uchar2(rgb0.c.b, rgb1.c.b);
}

//==============================================================================
// YUV444P16 KERNELS (4:4:4, 16-bit) - Professional HDR
//==============================================================================

/**
 * @brief YUV444 16-bit planar to RGB24 kernel
 */
__global__ void Yuv444P16ToRgb24Kernel(
    const uint8_t* __restrict__ pY,
    const uint8_t* __restrict__ pU,
    const uint8_t* __restrict__ pV,
    int nYuvPitch,
    uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    bool fullRange)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read 16-bit Y, U, V
    const uint16_t* pSrcY = reinterpret_cast<const uint16_t*>(pY + y * nYuvPitch) + x;
    const uint16_t* pSrcU = reinterpret_cast<const uint16_t*>(pU + y * nYuvPitch) + x;
    const uint16_t* pSrcV = reinterpret_cast<const uint16_t*>(pV + y * nYuvPitch) + x;
    
    ushort2 yy = *reinterpret_cast<const ushort2*>(pSrcY);
    ushort2 uu = *reinterpret_cast<const ushort2*>(pSrcU);
    ushort2 vv = *reinterpret_cast<const ushort2*>(pSrcV);
    
    // Convert
    RGB24 rgb0 = YuvToRgbForPixel<uint16_t>(yy.x, uu.x, vv.x, fullRange);
    RGB24 rgb1 = YuvToRgbForPixel<uint16_t>(yy.y, uu.y, vv.y, fullRange);
    
    // Vectorized write
    RGB24x2* pDst = reinterpret_cast<RGB24x2*>(pRgb + y * nRgbPitch + x * 3);
    *pDst = RGB24x2{rgb0.v, rgb1.v};
}

//==============================================================================
// PUBLIC API FUNCTIONS
//==============================================================================

/**
 * @brief Initialize color space conversion matrix
 * 
 * @param colorSpace Color space standard (1=BT.709, 6=BT.601, 9=BT.2020, etc.)
 * @param colorRange 0=Limited (TV), 1=Full (PC/JPEG)
 * @param stream CUDA stream
 */
void initColorSpaceMatrix(int colorSpace, int colorRange, cudaStream_t stream) {
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
}

// Backwards compatible overload
void initColorSpaceMatrix(int colorSpace, cudaStream_t stream) {
    SetMatYuv2Rgb(colorSpace, ColorRange_Limited, stream);
}

//------------------------------------------------------------------------------
// NV12 (4:2:0, 8-bit) Launch Functions
//------------------------------------------------------------------------------

/**
 * @brief Convert NV12 to RGB24 (packed)
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
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv12ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pNv12, nNv12Pitch, pRgb, nRgbPitch, nWidth, nHeight, nHeight, 
        colorRange == ColorRange_Full
    );
}

/**
 * @brief Convert NV12 to planar RGB (RGBP) for ML workflows
 * Output layout: [R plane][G plane][B plane], each H*W bytes
 */
void launchNv12ToRgbPlanar(
    const uint8_t* pNv12,
    int nNv12Pitch,
    uint8_t* pRgbp,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv12ToRgbPlanarKernel<<<gridDim, blockDim, 0, stream>>>(
        pNv12, nNv12Pitch, pRgbp, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

/**
 * @brief Convert NV12 to RGB24 (separate Y and UV planes)
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
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv12SeparateToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pUV, nYPitch, nUVPitch, pRgb, nRgbPitch, nWidth, nHeight,
        colorRange == ColorRange_Full
    );
}

//------------------------------------------------------------------------------
// P016 (4:2:0, 10/16-bit) Launch Functions
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
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    P016ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pP016, nP016Pitch, pRgb, nRgbPitch, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

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
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    P016ToRgbPlanarKernel<<<gridDim, blockDim, 0, stream>>>(
        pP016, nP016Pitch, pRgbp, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

//------------------------------------------------------------------------------
// NV16 (4:2:2, 8-bit) Launch Functions
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
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    // 4:2:2 - UV same height as Y, so only horizontal 2x1 processing
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv16ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pNv16, nNv16Pitch, pRgb, nRgbPitch, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

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
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv16ToRgbPlanarKernel<<<gridDim, blockDim, 0, stream>>>(
        pNv16, nNv16Pitch, pRgbp, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

//------------------------------------------------------------------------------
// P216 (4:2:2, 10/16-bit) Launch Functions
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
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    P216ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pP216, nP216Pitch, pRgb, nRgbPitch, nWidth, nHeight, nHeight,
        colorRange == ColorRange_Full
    );
}

//------------------------------------------------------------------------------
// YUV444 (4:4:4, 8-bit) Launch Functions
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
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Yuv444ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pU, pV, nYuvPitch, pRgb, nRgbPitch, nWidth, nHeight,
        colorRange == ColorRange_Full
    );
}

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
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Yuv444ToRgbPlanarKernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pU, pV, nYuvPitch, pRgbp, nWidth, nHeight,
        colorRange == ColorRange_Full
    );
}

//------------------------------------------------------------------------------
// YUV444P16 (4:4:4, 16-bit) Launch Functions
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
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 4);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Yuv444P16ToRgb24Kernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pU, pV, nYuvPitch, pRgb, nRgbPitch, nWidth, nHeight,
        colorRange == ColorRange_Full
    );
}

//==============================================================================
// LEGACY API COMPATIBILITY
// These functions maintain backwards compatibility with existing code
//==============================================================================

void launchNV12ToRGBKernel(
    const uint8_t* yPlane,
    uint8_t* rgbOutput,
    int width,
    int height,
    int yPitch,
    int rgbPitch,
    cudaStream_t stream)
{
    // Default to BT.709 for HD content, limited range
    launchNv12ToRgb24(yPlane, yPitch, rgbOutput, rgbPitch, width, height,
                      ColorSpaceStandard_BT709, ColorRange_Limited, stream);
}

void launchNV12ToRGBKernelWithUV(
    const uint8_t* yPlane,
    const uint8_t* uvPlane,
    uint8_t* rgbOutput,
    int width,
    int height,
    int yPitch,
    int uvPitch,
    int rgbPitch,
    cudaStream_t stream)
{
    launchNv12ToRgb24Separate(yPlane, uvPlane, yPitch, uvPitch, rgbOutput, rgbPitch,
                              width, height, ColorSpaceStandard_BT709, ColorRange_Limited, stream);
}

// Overloads without colorRange parameter (for backwards compatibility)
void launchNv12ToRgb24(
    const uint8_t* pNv12,
    int nNv12Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    cudaStream_t stream)
{
    launchNv12ToRgb24(pNv12, nNv12Pitch, pRgb, nRgbPitch, nWidth, nHeight,
                      colorSpace, ColorRange_Limited, stream);
}

void launchP016ToRgb24(
    const uint8_t* pP016,
    int nP016Pitch,
    uint8_t* pRgb,
    int nRgbPitch,
    int nWidth,
    int nHeight,
    int colorSpace,
    cudaStream_t stream)
{
    launchP016ToRgb24(pP016, nP016Pitch, pRgb, nRgbPitch, nWidth, nHeight,
                      colorSpace, ColorRange_Limited, stream);
}

//------------------------------------------------------------------------------
// ML-OPTIMIZED KERNELS (BCHW format with normalization)
//------------------------------------------------------------------------------

/**
 * @brief NV12 to BCHW float32 kernel with normalization
 * 
 * Output format: [B, C, H, W] where B=1, C=3 (RGB)
 * Normalization: (pixel / 255.0 - mean) / std
 * 
 * This is the optimal format for PyTorch ML models, eliminating the need for:
 * 1. HWC -> CHW permutation (expensive memory stride change)
 * 2. uint8 -> float32 conversion (separate kernel launch)
 * 3. Normalization (separate kernel launch)
 */
__global__ void Nv12ToBchwNormalizedKernel(
    const uint8_t* __restrict__ pY,
    const uint8_t* __restrict__ pUV,
    int nYPitch,
    int nUVPitch,
    float* __restrict__ pOutput,
    int nWidth,
    int nHeight,
    bool fullRange,
    float3 mean,      // Pre-computed: mean / 255.0
    float3 invStd)    // Pre-computed: 1.0 / (std * 255.0)
{
    // Each thread processes a 2x2 block of pixels
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Calculate output indices for BCHW format
    // Output layout: [B=0][C][H][W] = [0][C][y][x]
    // Stride: C * H * W
    const int planeSize = nWidth * nHeight;
    const int cStride = planeSize;
    
    // Read Y values
    const uint8_t* pSrcY = pY + y * nYPitch + x;
    uchar2 y0 = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 y1 = *reinterpret_cast<const uchar2*>(pSrcY + nYPitch);
    
    // Read UV
    const uint8_t* pSrcUV = pUV + (y / 2) * nUVPitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert 4 pixels
    RGB24 rgb00 = YuvToRgbForPixel<uint8_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint8_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint8_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint8_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Normalize and write to BCHW format
    // R plane (C=0)
    int idx0 = y * nWidth + x;
    int idx1 = (y + 1) * nWidth + x;
    
    pOutput[0 * cStride + idx0 + 0] = (rgb00.c.r * invStd.x) - mean.x;
    pOutput[0 * cStride + idx0 + 1] = (rgb01.c.r * invStd.x) - mean.x;
    pOutput[0 * cStride + idx1 + 0] = (rgb10.c.r * invStd.x) - mean.x;
    pOutput[0 * cStride + idx1 + 1] = (rgb11.c.r * invStd.x) - mean.x;
    
    // G plane (C=1)
    pOutput[1 * cStride + idx0 + 0] = (rgb00.c.g * invStd.y) - mean.y;
    pOutput[1 * cStride + idx0 + 1] = (rgb01.c.g * invStd.y) - mean.y;
    pOutput[1 * cStride + idx1 + 0] = (rgb10.c.g * invStd.y) - mean.y;
    pOutput[1 * cStride + idx1 + 1] = (rgb11.c.g * invStd.y) - mean.y;
    
    // B plane (C=2)
    pOutput[2 * cStride + idx0 + 0] = (rgb00.c.b * invStd.z) - mean.z;
    pOutput[2 * cStride + idx0 + 1] = (rgb01.c.b * invStd.z) - mean.z;
    pOutput[2 * cStride + idx1 + 0] = (rgb10.c.b * invStd.z) - mean.z;
    pOutput[2 * cStride + idx1 + 1] = (rgb11.c.b * invStd.z) - mean.z;
}

/**
 * @brief NV12 to BCHW float16 (half) kernel with normalization
 * 
 * Uses half precision (FP16) which provides:
 * - 2x memory bandwidth reduction vs float32
 * - 2x less memory usage
 * - Tensor Core acceleration on modern GPUs
 * - Sufficient precision for normalized uint8 data
 * 
 * Output format: [B, C, H, W] where B=1, C=3 (RGB), dtype=float16
 */
__global__ void Nv12ToBchwNormalizedFP16Kernel(
    const uint8_t* __restrict__ pY,
    const uint8_t* __restrict__ pUV,
    int nYPitch,
    int nUVPitch,
    half* __restrict__ pOutput,
    int nWidth,
    int nHeight,
    bool fullRange,
    float3 mean,      // Pre-computed: mean / 255.0
    float3 invStd)    // Pre-computed: 1.0 / (std * 255.0)
{
    // Each thread processes a 2x2 block of pixels
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Calculate output indices for BCHW format
    const int planeSize = nWidth * nHeight;
    const int cStride = planeSize;
    
    // Read Y values
    const uint8_t* pSrcY = pY + y * nYPitch + x;
    uchar2 y0 = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 y1 = *reinterpret_cast<const uchar2*>(pSrcY + nYPitch);
    
    // Read UV
    const uint8_t* pSrcUV = pUV + (y / 2) * nUVPitch + x;
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert 4 pixels
    RGB24 rgb00 = YuvToRgbForPixel<uint8_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint8_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint8_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint8_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Normalize and write to BCHW format as FP16
    int idx0 = y * nWidth + x;
    int idx1 = (y + 1) * nWidth + x;
    
    // R plane (C=0) - convert to half precision
    pOutput[0 * cStride + idx0 + 0] = __float2half((rgb00.c.r * invStd.x) - mean.x);
    pOutput[0 * cStride + idx0 + 1] = __float2half((rgb01.c.r * invStd.x) - mean.x);
    pOutput[0 * cStride + idx1 + 0] = __float2half((rgb10.c.r * invStd.x) - mean.x);
    pOutput[0 * cStride + idx1 + 1] = __float2half((rgb11.c.r * invStd.x) - mean.x);
    
    // G plane (C=1)
    pOutput[1 * cStride + idx0 + 0] = __float2half((rgb00.c.g * invStd.y) - mean.y);
    pOutput[1 * cStride + idx0 + 1] = __float2half((rgb01.c.g * invStd.y) - mean.y);
    pOutput[1 * cStride + idx1 + 0] = __float2half((rgb10.c.g * invStd.y) - mean.y);
    pOutput[1 * cStride + idx1 + 1] = __float2half((rgb11.c.g * invStd.y) - mean.y);
    
    // B plane (C=2)
    pOutput[2 * cStride + idx0 + 0] = __float2half((rgb00.c.b * invStd.z) - mean.z);
    pOutput[2 * cStride + idx0 + 1] = __float2half((rgb01.c.b * invStd.z) - mean.z);
    pOutput[2 * cStride + idx1 + 0] = __float2half((rgb10.c.b * invStd.z) - mean.z);
    pOutput[2 * cStride + idx1 + 1] = __float2half((rgb11.c.b * invStd.z) - mean.z);
}

/**
 * @brief Launch NV12 to BCHW normalized conversion
 * 
 * @param pY Y plane pointer (device)
 * @param pUV UV plane pointer (device)
 * @param nYPitch Y plane pitch
 * @param nUVPitch UV plane pitch
 * @param pOutput Output buffer [B, C, H, W] float32 (device)
 * @param nWidth Frame width
 * @param nHeight Frame height
 * @param colorSpace Color space standard
 * @param colorRange Color range (limited/full)
 * @param mean Pre-computed mean values (mean/255.0) for R, G, B
 * @param invStd Pre-computed inverse std values (1.0/(std*255.0)) for R, G, B
 * @param stream CUDA stream
 */
void launchNv12ToBchwNormalized(
    const uint8_t* pY,
    const uint8_t* pUV,
    int nYPitch,
    int nUVPitch,
    float* pOutput,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    float3 mean,
    float3 invStd,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv12ToBchwNormalizedKernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pUV, nYPitch, nUVPitch, pOutput, nWidth, nHeight,
        colorRange == ColorRange_Full, mean, invStd
    );
}

/**
 * @brief Launch NV12 to BCHW float16 (half) normalized conversion
 * 
 * @param pY Y plane pointer (device)
 * @param pUV UV plane pointer (device)
 * @param nYPitch Y plane pitch
 * @param nUVPitch UV plane pitch
 * @param pOutput Output buffer [B, C, H, W] float16 (device)
 * @param nWidth Frame width
 * @param nHeight Frame height
 * @param colorSpace Color space standard
 * @param colorRange Color range (limited/full)
 * @param mean Pre-computed mean values (mean/255.0) for R, G, B
 * @param invStd Pre-computed inverse std values (1.0/(std*255.0)) for R, G, B
 * @param stream CUDA stream
 */
void launchNv12ToBchwNormalizedFP16(
    const uint8_t* pY,
    const uint8_t* pUV,
    int nYPitch,
    int nUVPitch,
    half* pOutput,
    int nWidth,
    int nHeight,
    int colorSpace,
    int colorRange,
    float3 mean,
    float3 invStd,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Nv12ToBchwNormalizedFP16Kernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pUV, nYPitch, nUVPitch, pOutput, nWidth, nHeight,
        colorRange == ColorRange_Full, mean, invStd
    );
}

//------------------------------------------------------------------------------
// BATCH ML-OPTIMIZED KERNELS
//------------------------------------------------------------------------------

/**
 * @brief Process multiple frames in a batch with BCHW output
 * 
 * This kernel processes B frames in parallel, outputting [B, C, H, W]
 * Each block processes one frame, allowing efficient batch processing
 */
__global__ void Nv12BatchToBchwKernel(
    const uint8_t* __restrict__ pY[],      // Array of Y plane pointers
    const uint8_t* __restrict__ pUV[],     // Array of UV plane pointers
    const int* __restrict__ nYPitch,       // Array of Y pitches
    const int* __restrict__ nUVPitch,      // Array of UV pitches
    float* __restrict__ pOutput,           // Output [B, C, H, W]
    int nWidth,
    int nHeight,
    int batchSize,
    bool fullRange,
    float3 mean,
    float3 invStd)
{
    int frameIdx = blockIdx.z;  // Which frame in batch
    if (frameIdx >= batchSize) return;
    
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Calculate output indices for BCHW format
    const int frameStride = 3 * nWidth * nHeight;  // C * H * W
    const int cStride = nWidth * nHeight;
    float* pFrameOutput = pOutput + frameIdx * frameStride;
    
    // Read from this frame's input
    const uint8_t* pSrcY = pY[frameIdx] + y * nYPitch[frameIdx] + x;
    const uint8_t* pSrcUV = pUV[frameIdx] + (y / 2) * nUVPitch[frameIdx] + x;
    
    uchar2 y0 = *reinterpret_cast<const uchar2*>(pSrcY);
    uchar2 y1 = *reinterpret_cast<const uchar2*>(pSrcY + nYPitch[frameIdx]);
    uchar2 uv = *reinterpret_cast<const uchar2*>(pSrcUV);
    
    // Convert
    RGB24 rgb00 = YuvToRgbForPixel<uint8_t>(y0.x, uv.x, uv.y, fullRange);
    RGB24 rgb01 = YuvToRgbForPixel<uint8_t>(y0.y, uv.x, uv.y, fullRange);
    RGB24 rgb10 = YuvToRgbForPixel<uint8_t>(y1.x, uv.x, uv.y, fullRange);
    RGB24 rgb11 = YuvToRgbForPixel<uint8_t>(y1.y, uv.x, uv.y, fullRange);
    
    // Write normalized BCHW
    int idx0 = y * nWidth + x;
    int idx1 = (y + 1) * nWidth + x;
    
    // R plane
    pFrameOutput[0 * cStride + idx0 + 0] = (rgb00.c.r * invStd.x) - mean.x;
    pFrameOutput[0 * cStride + idx0 + 1] = (rgb01.c.r * invStd.x) - mean.x;
    pFrameOutput[0 * cStride + idx1 + 0] = (rgb10.c.r * invStd.x) - mean.x;
    pFrameOutput[0 * cStride + idx1 + 1] = (rgb11.c.r * invStd.x) - mean.x;
    
    // G plane
    pFrameOutput[1 * cStride + idx0 + 0] = (rgb00.c.g * invStd.y) - mean.y;
    pFrameOutput[1 * cStride + idx0 + 1] = (rgb01.c.g * invStd.y) - mean.y;
    pFrameOutput[1 * cStride + idx1 + 0] = (rgb10.c.g * invStd.y) - mean.y;
    pFrameOutput[1 * cStride + idx1 + 1] = (rgb11.c.g * invStd.y) - mean.y;
    
    // B plane
    pFrameOutput[2 * cStride + idx0 + 0] = (rgb00.c.b * invStd.z) - mean.z;
    pFrameOutput[2 * cStride + idx0 + 1] = (rgb01.c.b * invStd.z) - mean.z;
    pFrameOutput[2 * cStride + idx1 + 0] = (rgb10.c.b * invStd.z) - mean.z;
    pFrameOutput[2 * cStride + idx1 + 1] = (rgb11.c.b * invStd.z) - mean.z;
}

/**
 * @brief Launch batch NV12 to BCHW normalized conversion
 */
void launchNv12BatchToBchw(
    const uint8_t* pY[],
    const uint8_t* pUV[],
    const int nYPitch[],
    const int nUVPitch[],
    float* pOutput,
    int nWidth,
    int nHeight,
    int batchSize,
    int colorSpace,
    int colorRange,
    float3 mean,
    float3 invStd,
    cudaStream_t stream)
{
    SetMatYuv2Rgb(colorSpace, colorRange, stream);
    
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4,
        batchSize
    );
    
    Nv12BatchToBchwKernel<<<gridDim, blockDim, 0, stream>>>(
        pY, pUV, nYPitch, nUVPitch, pOutput, nWidth, nHeight, batchSize,
        colorRange == ColorRange_Full, mean, invStd
    );
}

//------------------------------------------------------------------------------
// UTILITY FUNCTIONS
//------------------------------------------------------------------------------

/**
 * @brief Pre-compute normalization constants for ML inference
 * 
 * Standard ImageNet normalization:
 *   mean = [0.485, 0.456, 0.406]
 *   std = [0.229, 0.224, 0.225]
 * 
 * This function converts to:
 *   mean_cuda = mean / 255.0
 *   invStd_cuda = 1.0 / (std * 255.0)
 * 
 * @param meanRGB Mean values for R, G, B (e.g., [0.485, 0.456, 0.406])
 * @param stdRGB Std values for R, G, B (e.g., [0.229, 0.224, 0.225])
 * @param meanOut Output pre-computed mean for CUDA kernel
 * @param invStdOut Output pre-computed inverse std for CUDA kernel
 */
void computeNormalizationConstants(
    const float meanRGB[3],
    const float stdRGB[3],
    float3* meanOut,
    float3* invStdOut)
{
    meanOut->x = meanRGB[0] / 255.0f;
    meanOut->y = meanRGB[1] / 255.0f;
    meanOut->z = meanRGB[2] / 255.0f;
    
    invStdOut->x = 1.0f / (stdRGB[0] * 255.0f);
    invStdOut->y = 1.0f / (stdRGB[1] * 255.0f);
    invStdOut->z = 1.0f / (stdRGB[2] * 255.0f);
}

//------------------------------------------------------------------------------
// UNIVERSAL RGB24 TO BCHW CONVERSION KERNELS
//------------------------------------------------------------------------------

/**
 * @brief Universal kernel: Convert RGB24 (HWC) to BCHW float32 with normalization
 * 
 * This kernel works with ANY input format that can be converted to RGB24 first.
 * Use this as a fallback for formats that don't have direct BCHW kernels.
 * 
 * Input: RGB24 interleaved [H, W, 3] uint8
 * Output: [B=1, C=3, H, W] float32
 */
__global__ void Rgb24ToBchwKernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    float* __restrict__ pOutput,
    int nWidth,
    int nHeight,
    float3 mean,
    float3 invStd)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    const int planeSize = nWidth * nHeight;
    const int cStride = planeSize;
    
    // Read 2x2 block of RGB24 pixels
    const uint8_t* pSrc0 = pRgb + y * nRgbPitch + x * 3;
    const uint8_t* pSrc1 = pRgb + (y + 1) * nRgbPitch + x * 3;
    
    // Pixel (x, y)
    uint8_t r00 = pSrc0[0];
    uint8_t g00 = pSrc0[1];
    uint8_t b00 = pSrc0[2];
    
    uint8_t r01 = pSrc0[3];
    uint8_t g01 = pSrc0[4];
    uint8_t b01 = pSrc0[5];
    
    uint8_t r10 = pSrc1[0];
    uint8_t g10 = pSrc1[1];
    uint8_t b10 = pSrc1[2];
    
    uint8_t r11 = pSrc1[3];
    uint8_t g11 = pSrc1[4];
    uint8_t b11 = pSrc1[5];
    
    // Write to BCHW format
    int idx0 = y * nWidth + x;
    int idx1 = (y + 1) * nWidth + x;
    
    // R plane
    pOutput[0 * cStride + idx0 + 0] = (r00 * invStd.x) - mean.x;
    pOutput[0 * cStride + idx0 + 1] = (r01 * invStd.x) - mean.x;
    pOutput[0 * cStride + idx1 + 0] = (r10 * invStd.x) - mean.x;
    pOutput[0 * cStride + idx1 + 1] = (r11 * invStd.x) - mean.x;
    
    // G plane
    pOutput[1 * cStride + idx0 + 0] = (g00 * invStd.y) - mean.y;
    pOutput[1 * cStride + idx0 + 1] = (g01 * invStd.y) - mean.y;
    pOutput[1 * cStride + idx1 + 0] = (g10 * invStd.y) - mean.y;
    pOutput[1 * cStride + idx1 + 1] = (g11 * invStd.y) - mean.y;
    
    // B plane
    pOutput[2 * cStride + idx0 + 0] = (b00 * invStd.z) - mean.z;
    pOutput[2 * cStride + idx0 + 1] = (b01 * invStd.z) - mean.z;
    pOutput[2 * cStride + idx1 + 0] = (b10 * invStd.z) - mean.z;
    pOutput[2 * cStride + idx1 + 1] = (b11 * invStd.z) - mean.z;
}

/**
 * @brief Universal kernel: Convert RGB24 (HWC) to BCHW float16 with normalization
 */
__global__ void Rgb24ToBchwFP16Kernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    half* __restrict__ pOutput,
    int nWidth,
    int nHeight,
    float3 mean,
    float3 invStd)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    const int planeSize = nWidth * nHeight;
    const int cStride = planeSize;
    
    // Read 2x2 block of RGB24 pixels
    const uint8_t* pSrc0 = pRgb + y * nRgbPitch + x * 3;
    const uint8_t* pSrc1 = pRgb + (y + 1) * nRgbPitch + x * 3;
    
    uint8_t r00 = pSrc0[0], g00 = pSrc0[1], b00 = pSrc0[2];
    uint8_t r01 = pSrc0[3], g01 = pSrc0[4], b01 = pSrc0[5];
    uint8_t r10 = pSrc1[0], g10 = pSrc1[1], b10 = pSrc1[2];
    uint8_t r11 = pSrc1[3], g11 = pSrc1[4], b11 = pSrc1[5];
    
    // Write to BCHW format as FP16
    int idx0 = y * nWidth + x;
    int idx1 = (y + 1) * nWidth + x;
    
    // R plane
    pOutput[0 * cStride + idx0 + 0] = __float2half((r00 * invStd.x) - mean.x);
    pOutput[0 * cStride + idx0 + 1] = __float2half((r01 * invStd.x) - mean.x);
    pOutput[0 * cStride + idx1 + 0] = __float2half((r10 * invStd.x) - mean.x);
    pOutput[0 * cStride + idx1 + 1] = __float2half((r11 * invStd.x) - mean.x);
    
    // G plane
    pOutput[1 * cStride + idx0 + 0] = __float2half((g00 * invStd.y) - mean.y);
    pOutput[1 * cStride + idx0 + 1] = __float2half((g01 * invStd.y) - mean.y);
    pOutput[1 * cStride + idx1 + 0] = __float2half((g10 * invStd.y) - mean.y);
    pOutput[1 * cStride + idx1 + 1] = __float2half((g11 * invStd.y) - mean.y);
    
    // B plane
    pOutput[2 * cStride + idx0 + 0] = __float2half((b00 * invStd.z) - mean.z);
    pOutput[2 * cStride + idx0 + 1] = __float2half((b01 * invStd.z) - mean.z);
    pOutput[2 * cStride + idx1 + 0] = __float2half((b10 * invStd.z) - mean.z);
    pOutput[2 * cStride + idx1 + 1] = __float2half((b11 * invStd.z) - mean.z);
}

/**
 * @brief Launch RGB24 to BCHW float32 conversion
 */
void launchRgb24ToBchw(
    const uint8_t* pRgb,
    int nRgbPitch,
    float* pOutput,
    int nWidth,
    int nHeight,
    float3 mean,
    float3 invStd,
    cudaStream_t stream)
{
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Rgb24ToBchwKernel<<<gridDim, blockDim, 0, stream>>>(
        pRgb, nRgbPitch, pOutput, nWidth, nHeight, mean, invStd
    );
}

/**
 * @brief Launch RGB24 to BCHW float16 conversion
 */
void launchRgb24ToBchwFP16(
    const uint8_t* pRgb,
    int nRgbPitch,
    half* pOutput,
    int nWidth,
    int nHeight,
    float3 mean,
    float3 invStd,
    cudaStream_t stream)
{
    dim3 blockDim(32, 2);
    dim3 gridDim(
        (nWidth + 63) / 64,
        (nHeight + 3) / 4
    );
    
    Rgb24ToBchwFP16Kernel<<<gridDim, blockDim, 0, stream>>>(
        pRgb, nRgbPitch, pOutput, nWidth, nHeight, mean, invStd
    );
}

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
