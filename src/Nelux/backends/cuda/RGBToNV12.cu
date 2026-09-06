/*
 * RGBToNV12.cu - CUDA kernels for RGB to YUV conversion (encoding path)
 * 
 * This is the inverse of NV12ToRGB.cu, providing GPU-accelerated
 * color space conversion for the NVENC encoding pipeline.
 * 
 * Supports:
 * - RGB24 to NV12 (8-bit 4:2:0) - Primary format for h264_nvenc/hevc_nvenc
 * - RGB24 to P010 (10-bit 4:2:0) - HDR content
 * - RGB24 to YUV444P (8-bit 4:4:4) - High quality, no chroma subsampling
 * - Multiple color standards: BT.601, BT.709, BT.2020
 * - Limited range (16-235) and full range (0-255) output
 * 
 * SPDX-License-Identifier: MIT
 */

#ifdef NELUX_ENABLE_CUDA

#include <cuda_runtime.h>
#include <cstdint>

namespace nelux::backends::cuda
{

//------------------------------------------------------------------------------
// Color space standards (matches FFmpeg AVCOL_SPC_* values)
//------------------------------------------------------------------------------
enum ColorSpaceStandard {
    ColorSpaceStandard_BT709 = 1,       // HD content (most common)
    ColorSpaceStandard_Unspecified = 2, // Will default to BT.709
    ColorSpaceStandard_BT601 = 6,       // SD content
    ColorSpaceStandard_BT2020 = 9       // HDR/UHD content
};

enum ColorRange {
    ColorRange_Limited = 0,  // Y: 16-235, UV: 16-240 (MPEG/TV range)
    ColorRange_Full = 1      // Y: 0-255, UV: 0-255 (JPEG/PC range)
};

//------------------------------------------------------------------------------
// Constant memory for RGB to YUV conversion matrix
//------------------------------------------------------------------------------
__constant__ float matRgb2Yuv[3][3];
__constant__ float yuvOffset[3];  // Y, U, V offsets after conversion

//------------------------------------------------------------------------------
// Pre-computed RGB to YUV conversion matrices
// 
// Formula for limited range (BT.709):
//   Y  = 0.2126*R + 0.7152*G + 0.0722*B
//   Cb = -0.1146*R - 0.3854*G + 0.5000*B
//   Cr = 0.5000*R - 0.4542*G - 0.0458*B
// 
// Then scale for limited range: Y' = 16 + 219*Y, UV' = 128 + 224*UV
//------------------------------------------------------------------------------

// BT.709 (HD content) - Limited Range output [16-235]
static const float kMatRgb2YuvBT709Limited[3][3] = {
    { 0.182586f,  0.614231f,  0.062007f},   // Y  = 16 + (0.2126*R + 0.7152*G + 0.0722*B) * 219/255
    {-0.100644f, -0.338572f,  0.439216f},   // Cb = 128 + (-0.1146*R - 0.3854*G + 0.5*B) * 224/255
    { 0.439216f, -0.398942f, -0.040274f}    // Cr = 128 + (0.5*R - 0.4542*G - 0.0458*B) * 224/255
};
static const float kOffsetBT709Limited[3] = {16.0f, 128.0f, 128.0f};

// BT.709 (HD content) - Full Range output [0-255]
static const float kMatRgb2YuvBT709Full[3][3] = {
    { 0.212600f,  0.715200f,  0.072200f},   // Y  = 0.2126*R + 0.7152*G + 0.0722*B
    {-0.114572f, -0.385428f,  0.500000f},   // Cb = 128 + (-0.1146*R - 0.3854*G + 0.5*B)
    { 0.500000f, -0.454153f, -0.045847f}    // Cr = 128 + (0.5*R - 0.4542*G - 0.0458*B)
};
static const float kOffsetBT709Full[3] = {0.0f, 128.0f, 128.0f};

// BT.601 (SD content) - Limited Range
static const float kMatRgb2YuvBT601Limited[3][3] = {
    { 0.256788f,  0.504129f,  0.097906f},   // Y
    {-0.148223f, -0.290993f,  0.439216f},   // Cb
    { 0.439216f, -0.367788f, -0.071427f}    // Cr
};
static const float kOffsetBT601Limited[3] = {16.0f, 128.0f, 128.0f};

// BT.601 (SD content) - Full Range
static const float kMatRgb2YuvBT601Full[3][3] = {
    { 0.299000f,  0.587000f,  0.114000f},   // Y
    {-0.168736f, -0.331264f,  0.500000f},   // Cb
    { 0.500000f, -0.418688f, -0.081312f}    // Cr
};
static const float kOffsetBT601Full[3] = {0.0f, 128.0f, 128.0f};

// BT.2020 (HDR/UHD content) - Limited Range
static const float kMatRgb2YuvBT2020Limited[3][3] = {
    { 0.225613f,  0.582282f,  0.050928f},   // Y
    {-0.122655f, -0.316560f,  0.439216f},   // Cb
    { 0.439216f, -0.403890f, -0.035326f}    // Cr
};
static const float kOffsetBT2020Limited[3] = {16.0f, 128.0f, 128.0f};

// BT.2020 (HDR/UHD content) - Full Range
static const float kMatRgb2YuvBT2020Full[3][3] = {
    { 0.262700f,  0.678000f,  0.059300f},   // Y
    {-0.139630f, -0.360370f,  0.500000f},   // Cb
    { 0.500000f, -0.459786f, -0.040214f}    // Cr
};
static const float kOffsetBT2020Full[3] = {0.0f, 128.0f, 128.0f};

//------------------------------------------------------------------------------
// Set the RGB to YUV conversion matrix in constant memory
//------------------------------------------------------------------------------
void SetMatRgb2Yuv(int iMatrix, int colorRange, cudaStream_t stream) {
    // Cache the last-uploaded (device, matrix, range): the encoder uses one
    // triplet for the whole session, so without this every frame pays 2x H2D
    // constant uploads that serialize the encode stream. Identical constants,
    // fewer uploads.
    //
    // The key MUST include the device: __constant__ storage is per-device, so
    // a (matrix, range)-only key would skip a needed upload when the same
    // thread encodes on a second GPU (or after a device reset) and silently
    // tint every frame with stale constants. A key mismatch always
    // re-uploads, so the failure mode of a stale key is one redundant upload,
    // never wrong pixels. Single submit thread per converter, so thread_local
    // needs no mutex (cf. the mutex-guarded decode-side cache in NV12ToRGB.cu
    // which is shared across threads).
    static thread_local int cachedDevice = -1;
    static thread_local int cachedMatrix = -1;
    static thread_local int cachedRange = -1;
    int curDevice = -1;
    if (cudaGetDevice(&curDevice) != cudaSuccess)
        curDevice = -1;
    if (curDevice == cachedDevice && iMatrix == cachedMatrix &&
        colorRange == cachedRange) {
        return;
    }
    const float (*mat)[3] = nullptr;
    const float *offset = nullptr;

    bool isFullRange = (colorRange == ColorRange_Full);
    
    switch (iMatrix) {
        case ColorSpaceStandard_BT709:
        case ColorSpaceStandard_Unspecified:
        default:
            mat = isFullRange ? kMatRgb2YuvBT709Full : kMatRgb2YuvBT709Limited;
            offset = isFullRange ? kOffsetBT709Full : kOffsetBT709Limited;
            break;
            
        case ColorSpaceStandard_BT601:
            mat = isFullRange ? kMatRgb2YuvBT601Full : kMatRgb2YuvBT601Limited;
            offset = isFullRange ? kOffsetBT601Full : kOffsetBT601Limited;
            break;
            
        case ColorSpaceStandard_BT2020:
            mat = isFullRange ? kMatRgb2YuvBT2020Full : kMatRgb2YuvBT2020Limited;
            offset = isFullRange ? kOffsetBT2020Full : kOffsetBT2020Limited;
            break;
    }
    
    cudaMemcpyToSymbolAsync(matRgb2Yuv, mat, sizeof(float) * 9, 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(yuvOffset, offset, sizeof(float) * 3, 0, cudaMemcpyHostToDevice, stream);
    cachedDevice = curDevice;
    cachedMatrix = iMatrix;
    cachedRange = colorRange;
}

//------------------------------------------------------------------------------
// Device helper functions
//------------------------------------------------------------------------------
template<class T>
__device__ __forceinline__ T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

/**
 * @brief Convert a single RGB pixel to YUV using constant memory matrix
 */
__device__ __forceinline__ void RgbToYuv(uint8_t r, uint8_t g, uint8_t b,
                                          float& y, float& u, float& v) {
    float rf = static_cast<float>(r);
    float gf = static_cast<float>(g);
    float bf = static_cast<float>(b);
    
    y = matRgb2Yuv[0][0] * rf + matRgb2Yuv[0][1] * gf + matRgb2Yuv[0][2] * bf + yuvOffset[0];
    u = matRgb2Yuv[1][0] * rf + matRgb2Yuv[1][1] * gf + matRgb2Yuv[1][2] * bf + yuvOffset[1];
    v = matRgb2Yuv[2][0] * rf + matRgb2Yuv[2][1] * gf + matRgb2Yuv[2][2] * bf + yuvOffset[2];
}

//==============================================================================
// RGB24 TO NV12 KERNEL (8-bit 4:2:0)
// Most common format for NVENC
//==============================================================================

/**
 * @brief RGB24 to NV12 kernel - processes 2x2 pixel blocks
 * 
 * NV12 layout:
 *   Y plane: W x H bytes (one Y per pixel)
 *   UV plane: W x H/2 bytes (interleaved U,V, one pair per 2x2 block)
 * 
 * Each thread processes a 2x2 block of RGB pixels and produces:
 *   - 4 Y values (one per pixel)
 *   - 1 UV pair (average of 4 pixels)
 */
__global__ void Rgb24ToNv12Kernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pNv12,
    int nNv12Pitch,
    int nWidth,
    int nHeight,
    int nSurfaceHeight)  // Y plane height in output buffer
{
    // Each thread handles a 2x2 block
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Read 4 RGB pixels (2x2 block)
    const uint8_t* pSrc00 = pRgb + y * nRgbPitch + x * 3;
    const uint8_t* pSrc01 = pSrc00 + 3;
    const uint8_t* pSrc10 = pRgb + (y + 1) * nRgbPitch + x * 3;
    const uint8_t* pSrc11 = pSrc10 + 3;
    
    // Convert each pixel to YUV
    float y00, u00, v00, y01, u01, v01;
    float y10, u10, v10, y11, u11, v11;
    
    RgbToYuv(pSrc00[0], pSrc00[1], pSrc00[2], y00, u00, v00);
    RgbToYuv(pSrc01[0], pSrc01[1], pSrc01[2], y01, u01, v01);
    RgbToYuv(pSrc10[0], pSrc10[1], pSrc10[2], y10, u10, v10);
    RgbToYuv(pSrc11[0], pSrc11[1], pSrc11[2], y11, u11, v11);
    
    // Write Y plane (4 values)
    uint8_t* pDstY0 = pNv12 + y * nNv12Pitch + x;
    uint8_t* pDstY1 = pNv12 + (y + 1) * nNv12Pitch + x;
    
    pDstY0[0] = static_cast<uint8_t>(Clamp(y00 + 0.5f, 0.0f, 255.0f));
    pDstY0[1] = static_cast<uint8_t>(Clamp(y01 + 0.5f, 0.0f, 255.0f));
    pDstY1[0] = static_cast<uint8_t>(Clamp(y10 + 0.5f, 0.0f, 255.0f));
    pDstY1[1] = static_cast<uint8_t>(Clamp(y11 + 0.5f, 0.0f, 255.0f));
    
    // Average UV for 2x2 block and write to UV plane
    float uAvg = (u00 + u01 + u10 + u11) * 0.25f;
    float vAvg = (v00 + v01 + v10 + v11) * 0.25f;
    
    uint8_t* pDstUV = pNv12 + nSurfaceHeight * nNv12Pitch + (y / 2) * nNv12Pitch + x;
    pDstUV[0] = static_cast<uint8_t>(Clamp(uAvg + 0.5f, 0.0f, 255.0f));
    pDstUV[1] = static_cast<uint8_t>(Clamp(vAvg + 0.5f, 0.0f, 255.0f));
}

//==============================================================================
// RGB24 TO P010 KERNEL (10-bit 4:2:0)
// Used for HDR content
//==============================================================================

/**
 * @brief RGB24 to P010 kernel
 * 
 * P010 layout (10-bit stored in 16-bit words, MSB aligned):
 *   Y plane: W x H uint16 (Y values in upper 10 bits)
 *   UV plane: W x H/2 uint16 (interleaved U,V)
 */
__global__ void Rgb24ToP010Kernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pP010,
    int nP010Pitch,
    int nWidth,
    int nHeight,
    int nSurfaceHeight)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Read 4 RGB pixels
    const uint8_t* pSrc00 = pRgb + y * nRgbPitch + x * 3;
    const uint8_t* pSrc01 = pSrc00 + 3;
    const uint8_t* pSrc10 = pRgb + (y + 1) * nRgbPitch + x * 3;
    const uint8_t* pSrc11 = pSrc10 + 3;
    
    float y00, u00, v00, y01, u01, v01;
    float y10, u10, v10, y11, u11, v11;
    
    RgbToYuv(pSrc00[0], pSrc00[1], pSrc00[2], y00, u00, v00);
    RgbToYuv(pSrc01[0], pSrc01[1], pSrc01[2], y01, u01, v01);
    RgbToYuv(pSrc10[0], pSrc10[1], pSrc10[2], y10, u10, v10);
    RgbToYuv(pSrc11[0], pSrc11[1], pSrc11[2], y11, u11, v11);
    
    // Scale 8-bit YUV to 10-bit (left-shifted by 6 for MSB alignment in 16-bit)
    // P010 stores 10-bit values in upper 10 bits of 16-bit word
    auto toP010 = [](float val) -> uint16_t {
        int v = static_cast<int>(Clamp(val + 0.5f, 0.0f, 255.0f));
        return static_cast<uint16_t>(v << 8);  // 8-bit to 16-bit MSB aligned
    };
    
    // Write Y plane (16-bit values)
    uint16_t* pDstY0 = reinterpret_cast<uint16_t*>(pP010 + y * nP010Pitch) + x;
    uint16_t* pDstY1 = reinterpret_cast<uint16_t*>(pP010 + (y + 1) * nP010Pitch) + x;
    
    pDstY0[0] = toP010(y00);
    pDstY0[1] = toP010(y01);
    pDstY1[0] = toP010(y10);
    pDstY1[1] = toP010(y11);
    
    // Average UV and write to UV plane
    float uAvg = (u00 + u01 + u10 + u11) * 0.25f;
    float vAvg = (v00 + v01 + v10 + v11) * 0.25f;
    
    uint16_t* pDstUV = reinterpret_cast<uint16_t*>(pP010 + nSurfaceHeight * nP010Pitch + (y / 2) * nP010Pitch) + x;
    pDstUV[0] = toP010(uAvg);
    pDstUV[1] = toP010(vAvg);
}

//==============================================================================
// RGB24 TO YUV444P KERNEL (8-bit 4:4:4)
// No chroma subsampling - highest quality
//==============================================================================

/**
 * @brief RGB24 to YUV444P kernel
 * 
 * YUV444P layout (planar):
 *   Y plane: W x H bytes
 *   U plane: W x H bytes
 *   V plane: W x H bytes
 */
__global__ void Rgb24ToYuv444Kernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pY,
    uint8_t* __restrict__ pU,
    uint8_t* __restrict__ pV,
    int nYuvPitch,
    int nWidth,
    int nHeight)
{
    // Each thread processes 2 horizontal pixels
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read 2 RGB pixels
    const uint8_t* pSrc0 = pRgb + y * nRgbPitch + x * 3;
    const uint8_t* pSrc1 = pSrc0 + 3;
    
    float y0, u0, v0, y1, u1, v1;
    RgbToYuv(pSrc0[0], pSrc0[1], pSrc0[2], y0, u0, v0);
    RgbToYuv(pSrc1[0], pSrc1[1], pSrc1[2], y1, u1, v1);
    
    // Write to planar output (each pixel gets its own Y, U, V)
    uint8_t* pDstY = pY + y * nYuvPitch + x;
    uint8_t* pDstU = pU + y * nYuvPitch + x;
    uint8_t* pDstV = pV + y * nYuvPitch + x;
    
    pDstY[0] = static_cast<uint8_t>(Clamp(y0 + 0.5f, 0.0f, 255.0f));
    pDstY[1] = static_cast<uint8_t>(Clamp(y1 + 0.5f, 0.0f, 255.0f));
    pDstU[0] = static_cast<uint8_t>(Clamp(u0 + 0.5f, 0.0f, 255.0f));
    pDstU[1] = static_cast<uint8_t>(Clamp(u1 + 0.5f, 0.0f, 255.0f));
    pDstV[0] = static_cast<uint8_t>(Clamp(v0 + 0.5f, 0.0f, 255.0f));
    pDstV[1] = static_cast<uint8_t>(Clamp(v1 + 0.5f, 0.0f, 255.0f));
}

//==============================================================================
// RGB24 TO NV16 KERNEL (8-bit 4:2:2)
// Professional video format
//==============================================================================

/**
 * @brief RGB24 to NV16 kernel
 * 
 * NV16 layout (4:2:2 semi-planar):
 *   Y plane: W x H bytes
 *   UV plane: W x H bytes (interleaved U,V, one pair per 2 horizontal pixels)
 */
__global__ void Rgb24ToNv16Kernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pNv16,
    int nNv16Pitch,
    int nWidth,
    int nHeight,
    int nSurfaceHeight)
{
    // Each thread processes 2 horizontal pixels
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read 2 RGB pixels
    const uint8_t* pSrc0 = pRgb + y * nRgbPitch + x * 3;
    const uint8_t* pSrc1 = pSrc0 + 3;
    
    float y0, u0, v0, y1, u1, v1;
    RgbToYuv(pSrc0[0], pSrc0[1], pSrc0[2], y0, u0, v0);
    RgbToYuv(pSrc1[0], pSrc1[1], pSrc1[2], y1, u1, v1);
    
    // Write Y values
    uint8_t* pDstY = pNv16 + y * nNv16Pitch + x;
    pDstY[0] = static_cast<uint8_t>(Clamp(y0 + 0.5f, 0.0f, 255.0f));
    pDstY[1] = static_cast<uint8_t>(Clamp(y1 + 0.5f, 0.0f, 255.0f));
    
    // Average UV for 2 horizontal pixels (4:2:2 subsampling)
    float uAvg = (u0 + u1) * 0.5f;
    float vAvg = (v0 + v1) * 0.5f;
    
    // UV plane starts after Y plane
    uint8_t* pDstUV = pNv16 + nSurfaceHeight * nNv16Pitch + y * nNv16Pitch + x;
    pDstUV[0] = static_cast<uint8_t>(Clamp(uAvg + 0.5f, 0.0f, 255.0f));
    pDstUV[1] = static_cast<uint8_t>(Clamp(vAvg + 0.5f, 0.0f, 255.0f));
}

//==============================================================================
// RGB24 TO YUV420P KERNEL (8-bit 4:2:0 planar)
// Planar YUV format with separate U and V planes
//==============================================================================

/**
 * @brief RGB24 to YUV420P (I420) kernel
 * 
 * YUV420P layout (planar 4:2:0):
 *   Y plane: W x H bytes (one Y per pixel)
 *   U plane: W/2 x H/2 bytes (one U per 2x2 block)
 *   V plane: W/2 x H/2 bytes (one V per 2x2 block)
 * 
 * Each thread processes a 2x2 block of RGB pixels and produces:
 *   - 4 Y values (one per pixel)
 *   - 1 U value (average of 4 pixels)
 *   - 1 V value (average of 4 pixels)
 */
__global__ void Rgb24ToYuv420pKernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pY,
    uint8_t* __restrict__ pU,
    uint8_t* __restrict__ pV,
    int nYPitch,
    int nWidth,
    int nHeight)
{
    // Each thread handles a 2x2 block
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Read 4 RGB pixels (2x2 block)
    const uint8_t* pSrc00 = pRgb + y * nRgbPitch + x * 3;
    const uint8_t* pSrc01 = pSrc00 + 3;
    const uint8_t* pSrc10 = pRgb + (y + 1) * nRgbPitch + x * 3;
    const uint8_t* pSrc11 = pSrc10 + 3;
    
    // Convert each pixel to YUV
    float y00, u00, v00, y01, u01, v01;
    float y10, u10, v10, y11, u11, v11;
    
    RgbToYuv(pSrc00[0], pSrc00[1], pSrc00[2], y00, u00, v00);
    RgbToYuv(pSrc01[0], pSrc01[1], pSrc01[2], y01, u01, v01);
    RgbToYuv(pSrc10[0], pSrc10[1], pSrc10[2], y10, u10, v10);
    RgbToYuv(pSrc11[0], pSrc11[1], pSrc11[2], y11, u11, v11);
    
    // Write Y plane (4 values)
    uint8_t* pDstY0 = pY + y * nYPitch + x;
    uint8_t* pDstY1 = pY + (y + 1) * nYPitch + x;
    
    pDstY0[0] = static_cast<uint8_t>(Clamp(y00 + 0.5f, 0.0f, 255.0f));
    pDstY0[1] = static_cast<uint8_t>(Clamp(y01 + 0.5f, 0.0f, 255.0f));
    pDstY1[0] = static_cast<uint8_t>(Clamp(y10 + 0.5f, 0.0f, 255.0f));
    pDstY1[1] = static_cast<uint8_t>(Clamp(y11 + 0.5f, 0.0f, 255.0f));
    
    // Average UV for 2x2 block and write to separate U and V planes
    float uAvg = (u00 + u01 + u10 + u11) * 0.25f;
    float vAvg = (v00 + v01 + v10 + v11) * 0.25f;
    
    // U and V planes are half resolution
    int uvX = x / 2;
    int uvY = y / 2;
    int uvPitch = nYPitch / 2;  // U/V planes have half the width
    
    pU[uvY * uvPitch + uvX] = static_cast<uint8_t>(Clamp(uAvg + 0.5f, 0.0f, 255.0f));
    pV[uvY * uvPitch + uvX] = static_cast<uint8_t>(Clamp(vAvg + 0.5f, 0.0f, 255.0f));
}

//==============================================================================
// RGB24 TO P016/YUV420P16 KERNEL (16-bit 4:2:0 planar)
// 16-bit planar YUV format for high bit depth encoding
//==============================================================================

/**
 * @brief RGB24 to P016/YUV420P16 (16-bit planar 4:2:0) kernel
 * 
 * Output layout (16-bit per component):
 *   Y plane: W x H uint16 (MSB aligned)
 *   U plane: W/2 x H/2 uint16
 *   V plane: W/2 x H/2 uint16
 */
__global__ void Rgb24ToYuv420p16Kernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pY,
    uint8_t* __restrict__ pU,
    uint8_t* __restrict__ pV,
    int nYPitch,
    int nWidth,
    int nHeight)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }
    
    // Read 4 RGB pixels
    const uint8_t* pSrc00 = pRgb + y * nRgbPitch + x * 3;
    const uint8_t* pSrc01 = pSrc00 + 3;
    const uint8_t* pSrc10 = pRgb + (y + 1) * nRgbPitch + x * 3;
    const uint8_t* pSrc11 = pSrc10 + 3;
    
    float y00, u00, v00, y01, u01, v01;
    float y10, u10, v10, y11, u11, v11;
    
    RgbToYuv(pSrc00[0], pSrc00[1], pSrc00[2], y00, u00, v00);
    RgbToYuv(pSrc01[0], pSrc01[1], pSrc01[2], y01, u01, v01);
    RgbToYuv(pSrc10[0], pSrc10[1], pSrc10[2], y10, u10, v10);
    RgbToYuv(pSrc11[0], pSrc11[1], pSrc11[2], y11, u11, v11);
    
    // Convert 8-bit YUV to 16-bit (left-shifted by 8 for MSB alignment)
    auto toP016 = [](float val) -> uint16_t {
        int v = static_cast<int>(Clamp(val + 0.5f, 0.0f, 255.0f));
        return static_cast<uint16_t>(v << 8);
    };
    
    // Write Y plane (16-bit values)
    uint16_t* pDstY0 = reinterpret_cast<uint16_t*>(pY + y * nYPitch) + x;
    uint16_t* pDstY1 = reinterpret_cast<uint16_t*>(pY + (y + 1) * nYPitch) + x;
    
    pDstY0[0] = toP016(y00);
    pDstY0[1] = toP016(y01);
    pDstY1[0] = toP016(y10);
    pDstY1[1] = toP016(y11);
    
    // Average UV and write to separate planes
    float uAvg = (u00 + u01 + u10 + u11) * 0.25f;
    float vAvg = (v00 + v01 + v10 + v11) * 0.25f;
    
    int uvX = x / 2;
    int uvY = y / 2;
    int uvPitch = nYPitch / 2;
    
    uint16_t* pDstU = reinterpret_cast<uint16_t*>(pU + uvY * uvPitch) + uvX;
    uint16_t* pDstV = reinterpret_cast<uint16_t*>(pV + uvY * uvPitch) + uvX;
    
    pDstU[0] = toP016(uAvg);
    pDstV[0] = toP016(vAvg);
}

//==============================================================================
// HOST WRAPPER FUNCTIONS
//==============================================================================

/**
 * @brief Convert RGB24 to NV12 on GPU
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
    cudaStream_t stream)
{
    // Set conversion matrix
    SetMatRgb2Yuv(colorSpace, colorRange, stream);
    
    // Launch kernel - each thread handles 2x2 block
    dim3 blockSize(16, 16);
    dim3 gridSize((nWidth / 2 + blockSize.x - 1) / blockSize.x,
                  (nHeight / 2 + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToNv12Kernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pNv12, nNv12Pitch, nWidth, nHeight, nSurfaceHeight);
}

/**
 * @brief Convert RGB24 to P010 on GPU
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
    cudaStream_t stream)
{
    SetMatRgb2Yuv(colorSpace, colorRange, stream);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((nWidth / 2 + blockSize.x - 1) / blockSize.x,
                  (nHeight / 2 + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToP010Kernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pP010, nP010Pitch, nWidth, nHeight, nSurfaceHeight);
}

/**
 * @brief Convert RGB24 to YUV444P on GPU
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
    cudaStream_t stream)
{
    SetMatRgb2Yuv(colorSpace, colorRange, stream);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((nWidth / 2 + blockSize.x - 1) / blockSize.x,
                  (nHeight + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToYuv444Kernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pY, pU, pV, nYuvPitch, nWidth, nHeight);
}

/**
 * @brief Convert RGB24 to YUV420P (I420) on GPU
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
    cudaStream_t stream)
{
    SetMatRgb2Yuv(colorSpace, colorRange, stream);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((nWidth / 2 + blockSize.x - 1) / blockSize.x,
                  (nHeight / 2 + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToYuv420pKernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pY, pU, pV, nYPitch, nWidth, nHeight);
}

/**
 * @brief Convert RGB24 to YUV420P16 (P016 planar) on GPU
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
    cudaStream_t stream)
{
    SetMatRgb2Yuv(colorSpace, colorRange, stream);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((nWidth / 2 + blockSize.x - 1) / blockSize.x,
                  (nHeight / 2 + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToYuv420p16Kernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pY, pU, pV, nYPitch, nWidth, nHeight);
}

/**
 * @brief Convert RGB24 to NV16 on GPU
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
    cudaStream_t stream)
{
    SetMatRgb2Yuv(colorSpace, colorRange, stream);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((nWidth / 2 + blockSize.x - 1) / blockSize.x,
                  (nHeight + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToNv16Kernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pNv16, nNv16Pitch, nWidth, nHeight, nSurfaceHeight);
}

//==============================================================================
// RGB24 TO PACKED RGB/BGR KERNELS (for NVENC RGB input)
// NVENC supports BGR0/BGRA/RGB0/RGBA directly without YUV conversion
//==============================================================================

/**
 * @brief RGB24 to BGR0/BGRA kernel (swaps R and B channels)
 * 
 * BGR0 layout: B G R (3 bytes per pixel, no alpha)
 * BGRA layout: B G R A (4 bytes per pixel)
 */
__global__ void Rgb24ToBgrKernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pBgr,
    int nBgrPitch,
    int nWidth,
    int nHeight,
    bool addAlpha)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read RGB pixel
    const uint8_t* pSrc = pRgb + y * nRgbPitch + x * 3;
    uint8_t r = pSrc[0];
    uint8_t g = pSrc[1];
    uint8_t b = pSrc[2];
    
    // Write BGR (swapped)
    uint8_t* pDst = pBgr + y * nBgrPitch + x * (addAlpha ? 4 : 3);
    pDst[0] = b;  // B
    pDst[1] = g;  // G
    pDst[2] = r;  // R
    
    if (addAlpha) {
        pDst[3] = 255;  // A (opaque)
    }
}

/**
 * @brief RGB24 to RGB0/RGBA kernel (packed, no channel swap)
 */
__global__ void Rgb24ToRgbPackedKernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pDst,
    int nDstPitch,
    int nWidth,
    int nHeight,
    bool addAlpha)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read RGB pixel
    const uint8_t* pSrc = pRgb + y * nRgbPitch + x * 3;
    
    // Write RGB (same order, optionally with alpha)
    uint8_t* pDstPixel = pDst + y * nDstPitch + x * (addAlpha ? 4 : 3);
    pDstPixel[0] = pSrc[0];  // R
    pDstPixel[1] = pSrc[1];  // G
    pDstPixel[2] = pSrc[2];  // B
    
    if (addAlpha) {
        pDstPixel[3] = 255;  // A (opaque)
    }
}

//==============================================================================
// RGB24 TO GBRP KERNEL (planar RGB for NVENC)
// GBRP layout: G plane, B plane, R plane (FFmpeg convention)
//==============================================================================

/**
 * @brief RGB24 to GBRP (planar GBR) kernel
 * 
 * Output layout (planar):
 *   G plane: W x H
 *   B plane: W x H  
 *   R plane: W x H
 */
__global__ void Rgb24ToGbrpKernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pG,
    uint8_t* __restrict__ pB,
    uint8_t* __restrict__ pR,
    int nPitch,
    int nWidth,
    int nHeight)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read RGB pixel
    const uint8_t* pSrc = pRgb + y * nRgbPitch + x * 3;
    uint8_t r = pSrc[0];
    uint8_t g = pSrc[1];
    uint8_t b = pSrc[2];
    
    // Write to separate planes (FFmpeg GBRP order: G, B, R)
    int idx = y * nPitch + x;
    pG[idx] = g;
    pB[idx] = b;
    pR[idx] = r;
}

//==============================================================================
// RGB48 (16-bit RGB) TO GBRP16LE KERNEL
//==============================================================================

/**
 * @brief RGB48 (16-bit per channel) to GBRP16LE kernel
 * 
 * Input: Packed RGB48 (R16 G16 B16 = 6 bytes per pixel)
 * Output: Planar GBR 16-bit (G plane, B plane, R plane)
 */
__global__ void Rgb48ToGbrp16Kernel(
    const uint8_t* __restrict__ pRgb48,
    int nRgbPitch,
    uint8_t* __restrict__ pG,
    uint8_t* __restrict__ pB,
    uint8_t* __restrict__ pR,
    int nPitch,
    int nWidth,
    int nHeight)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read RGB48 pixel (16-bit per channel, little-endian)
    const uint16_t* pSrc = reinterpret_cast<const uint16_t*>(pRgb48 + y * nRgbPitch + x * 6);
    uint16_t r = pSrc[0];
    uint16_t g = pSrc[1];
    uint16_t b = pSrc[2];
    
    // Write to separate 16-bit planes (FFmpeg GBRP16LE order: G, B, R)
    uint16_t* pG16 = reinterpret_cast<uint16_t*>(pG + y * nPitch) + x;
    uint16_t* pB16 = reinterpret_cast<uint16_t*>(pB + y * nPitch) + x;
    uint16_t* pR16 = reinterpret_cast<uint16_t*>(pR + y * nPitch) + x;
    
    *pG16 = g;
    *pB16 = b;
    *pR16 = r;
}

//==============================================================================
// P210/P216 KERNELS (10/16-bit 4:2:2 for NVENC)
//==============================================================================

/**
 * @brief RGB24 to P210 (10-bit 4:2:2) kernel
 * 
 * P210 layout (similar to P010 but 4:2:2):
 *   Y plane: W x H uint16 (10-bit in upper bits)
 *   UV plane: W x H uint16 (interleaved, 10-bit)
 */
__global__ void Rgb24ToP210Kernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pP210,
    int nP210Pitch,
    int nWidth,
    int nHeight,
    int nSurfaceHeight)
{
    // Each thread processes 2 horizontal pixels (4:2:2 chroma subsampling)
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read 2 RGB pixels
    const uint8_t* pSrc0 = pRgb + y * nRgbPitch + x * 3;
    const uint8_t* pSrc1 = pSrc0 + 3;
    
    float y0, u0, v0, y1, u1, v1;
    RgbToYuv(pSrc0[0], pSrc0[1], pSrc0[2], y0, u0, v0);
    RgbToYuv(pSrc1[0], pSrc1[1], pSrc1[2], y1, u1, v1);
    
    // Convert 8-bit to 10-bit MSB-aligned in a 16-bit word: expand 8->10 bit
    // (<<2) then MSB-align the 10-bit value in 16 bits (<<6) = <<8. This matches
    // the P010 (4:2:0 10-bit) and P216 siblings; the previous <<6 left the top
    // two bits zero, capping output at ~25% of full 10-bit scale.
    auto toP210 = [](float val) -> uint16_t {
        int v = static_cast<int>(Clamp(val + 0.5f, 0.0f, 255.0f));
        return static_cast<uint16_t>(v << 8);  // 8-bit -> 10-bit MSB aligned
    };
    
    // Write Y plane (16-bit values)
    uint16_t* pDstY = reinterpret_cast<uint16_t*>(pP210 + y * nP210Pitch) + x;
    pDstY[0] = toP210(y0);
    pDstY[1] = toP210(y1);
    
    // Average UV for 2 horizontal pixels (4:2:2 subsampling)
    float uAvg = (u0 + u1) * 0.5f;
    float vAvg = (v0 + v1) * 0.5f;
    
    // UV plane (same height as Y for 4:2:2)
    uint16_t* pDstUV = reinterpret_cast<uint16_t*>(pP210 + nSurfaceHeight * nP210Pitch + y * nP210Pitch) + x;
    pDstUV[0] = toP210(uAvg);
    pDstUV[1] = toP210(vAvg);
}

/**
 * @brief RGB24 to P216 (16-bit 4:2:2) kernel
 * 
 * P216 layout:
 *   Y plane: W x H uint16 (MSB aligned)
 *   UV plane: W x H uint16 (interleaved, MSB aligned)
 */
__global__ void Rgb24ToP216Kernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pP216,
    int nP216Pitch,
    int nWidth,
    int nHeight,
    int nSurfaceHeight)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x + 1 >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read 2 RGB pixels
    const uint8_t* pSrc0 = pRgb + y * nRgbPitch + x * 3;
    const uint8_t* pSrc1 = pSrc0 + 3;
    
    float y0, u0, v0, y1, u1, v1;
    RgbToYuv(pSrc0[0], pSrc0[1], pSrc0[2], y0, u0, v0);
    RgbToYuv(pSrc1[0], pSrc1[1], pSrc1[2], y1, u1, v1);
    
    // Convert to 16-bit (left-shifted by 8 for MSB alignment)
    auto toP216 = [](float val) -> uint16_t {
        int v = static_cast<int>(Clamp(val + 0.5f, 0.0f, 255.0f));
        return static_cast<uint16_t>(v << 8);
    };
    
    // Write Y plane
    uint16_t* pDstY = reinterpret_cast<uint16_t*>(pP216 + y * nP216Pitch) + x;
    pDstY[0] = toP216(y0);
    pDstY[1] = toP216(y1);
    
    // Average UV
    float uAvg = (u0 + u1) * 0.5f;
    float vAvg = (v0 + v1) * 0.5f;
    
    // UV plane
    uint16_t* pDstUV = reinterpret_cast<uint16_t*>(pP216 + nSurfaceHeight * nP216Pitch + y * nP216Pitch) + x;
    pDstUV[0] = toP216(uAvg);
    pDstUV[1] = toP216(vAvg);
}

//==============================================================================
// YUV444 10/16-bit ENCODE KERNELS
//==============================================================================

/**
 * @brief RGB24 to YUV444P10LE (10-bit planar 4:4:4) kernel
 */
__global__ void Rgb24ToYuv444P10Kernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pY,
    uint8_t* __restrict__ pU,
    uint8_t* __restrict__ pV,
    int nYuvPitch,
    int nWidth,
    int nHeight)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read RGB pixel
    const uint8_t* pSrc = pRgb + y * nRgbPitch + x * 3;
    
    float yf, uf, vf;
    RgbToYuv(pSrc[0], pSrc[1], pSrc[2], yf, uf, vf);
    
    // Convert to 10-bit
    auto to10bit = [](float val) -> uint16_t {
        int v = static_cast<int>(Clamp(val + 0.5f, 0.0f, 255.0f));
        return static_cast<uint16_t>(v << 2);  // 8-bit to 10-bit
    };
    
    // Write to 16-bit planes
    uint16_t* pDstY = reinterpret_cast<uint16_t*>(pY + y * nYuvPitch) + x;
    uint16_t* pDstU = reinterpret_cast<uint16_t*>(pU + y * nYuvPitch) + x;
    uint16_t* pDstV = reinterpret_cast<uint16_t*>(pV + y * nYuvPitch) + x;
    
    *pDstY = to10bit(yf);
    *pDstU = to10bit(uf);
    *pDstV = to10bit(vf);
}

/**
 * @brief RGB24 to YUV444P16LE (16-bit planar 4:4:4) kernel
 */
__global__ void Rgb24ToYuv444P16Kernel(
    const uint8_t* __restrict__ pRgb,
    int nRgbPitch,
    uint8_t* __restrict__ pY,
    uint8_t* __restrict__ pU,
    uint8_t* __restrict__ pV,
    int nYuvPitch,
    int nWidth,
    int nHeight)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x >= nWidth || y >= nHeight) {
        return;
    }
    
    // Read RGB pixel
    const uint8_t* pSrc = pRgb + y * nRgbPitch + x * 3;
    
    float yf, uf, vf;
    RgbToYuv(pSrc[0], pSrc[1], pSrc[2], yf, uf, vf);
    
    // Convert to 16-bit
    auto to16bit = [](float val) -> uint16_t {
        int v = static_cast<int>(Clamp(val + 0.5f, 0.0f, 255.0f));
        return static_cast<uint16_t>(v << 8);  // 8-bit to 16-bit MSB aligned
    };
    
    // Write to 16-bit planes
    uint16_t* pDstY = reinterpret_cast<uint16_t*>(pY + y * nYuvPitch) + x;
    uint16_t* pDstU = reinterpret_cast<uint16_t*>(pU + y * nYuvPitch) + x;
    uint16_t* pDstV = reinterpret_cast<uint16_t*>(pV + y * nYuvPitch) + x;
    
    *pDstY = to16bit(yf);
    *pDstU = to16bit(uf);
    *pDstV = to16bit(vf);
}

//==============================================================================
// HOST WRAPPER FUNCTIONS FOR NEW FORMATS
//==============================================================================

/**
 * @brief Convert RGB24 to BGR0/BGRA (packed BGR for NVENC)
 */
void Rgb24ToBgr(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pBgr,
    int nBgrPitch,
    int nWidth,
    int nHeight,
    bool addAlpha,
    cudaStream_t stream)
{
    // No color matrix needed for channel swap
    dim3 blockSize(32, 8);
    dim3 gridSize((nWidth + blockSize.x - 1) / blockSize.x,
                  (nHeight + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToBgrKernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pBgr, nBgrPitch, nWidth, nHeight, addAlpha);
}

/**
 * @brief Convert RGB24 to RGB0/RGBA (packed RGB for NVENC)
 */
void Rgb24ToRgbPacked(
    const uint8_t* pRgb,
    int nRgbPitch,
    uint8_t* pDst,
    int nDstPitch,
    int nWidth,
    int nHeight,
    bool addAlpha,
    cudaStream_t stream)
{
    dim3 blockSize(32, 8);
    dim3 gridSize((nWidth + blockSize.x - 1) / blockSize.x,
                  (nHeight + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToRgbPackedKernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pDst, nDstPitch, nWidth, nHeight, addAlpha);
}

/**
 * @brief Convert RGB24 to GBRP (planar GBR for NVENC)
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
    cudaStream_t stream)
{
    dim3 blockSize(32, 8);
    dim3 gridSize((nWidth + blockSize.x - 1) / blockSize.x,
                  (nHeight + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToGbrpKernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pG, pB, pR, nPitch, nWidth, nHeight);
}

/**
 * @brief Convert RGB48 to GBRP16LE (planar 16-bit GBR)
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
    cudaStream_t stream)
{
    dim3 blockSize(32, 8);
    dim3 gridSize((nWidth + blockSize.x - 1) / blockSize.x,
                  (nHeight + blockSize.y - 1) / blockSize.y);
    
    Rgb48ToGbrp16Kernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb48, nRgbPitch, pG, pB, pR, nPitch, nWidth, nHeight);
}

/**
 * @brief Convert RGB24 to P210 (10-bit 4:2:2)
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
    cudaStream_t stream)
{
    SetMatRgb2Yuv(colorSpace, colorRange, stream);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((nWidth / 2 + blockSize.x - 1) / blockSize.x,
                  (nHeight + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToP210Kernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pP210, nP210Pitch, nWidth, nHeight, nSurfaceHeight);
}

/**
 * @brief Convert RGB24 to P216 (16-bit 4:2:2)
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
    cudaStream_t stream)
{
    SetMatRgb2Yuv(colorSpace, colorRange, stream);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((nWidth / 2 + blockSize.x - 1) / blockSize.x,
                  (nHeight + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToP216Kernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pP216, nP216Pitch, nWidth, nHeight, nSurfaceHeight);
}

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
    cudaStream_t stream)
{
    SetMatRgb2Yuv(colorSpace, colorRange, stream);
    
    dim3 blockSize(32, 8);
    dim3 gridSize((nWidth + blockSize.x - 1) / blockSize.x,
                  (nHeight + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToYuv444P10Kernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pY, pU, pV, nYuvPitch, nWidth, nHeight);
}

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
    cudaStream_t stream)
{
    SetMatRgb2Yuv(colorSpace, colorRange, stream);
    
    dim3 blockSize(32, 8);
    dim3 gridSize((nWidth + blockSize.x - 1) / blockSize.x,
                  (nHeight + blockSize.y - 1) / blockSize.y);
    
    Rgb24ToYuv444P16Kernel<<<gridSize, blockSize, 0, stream>>>(
        pRgb, nRgbPitch, pY, pU, pV, nYuvPitch, nWidth, nHeight);
}

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
