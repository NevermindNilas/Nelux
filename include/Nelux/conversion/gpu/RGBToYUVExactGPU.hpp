/*
 * RGBToYUVExactGPU.hpp - host side of the byte-exact CUDA RGB->YUV
 * kernels (RGBToYUVExact.cu).
 *
 * The kernels reproduce what RGBToAutoConverter's legacy swscale context
 * (sws_getContext(..., SWS_BILINEAR) + sws_setColorspaceDetails, same size,
 * libswscale 10.1.102 on x86-64) writes, byte for byte. Everything swscale
 * derives at context-build time is derived here the same way:
 *
 *  - the RGB->YUV integer table: a port of fill_rgb2yuv_table (utils.c),
 *    including its hard-coded BT.601 special case and the forced
 *    limited-range table (full range is a separate pass, below);
 *  - the full-range pass constants: a port of init_range_convert_constants
 *    / solve_range_convert (swscale.c), truncated to the uint16 coeff /
 *    int32 offset that lumRangeToJpeg / chrRangeToJpeg actually use;
 *  - the chroma filters: a port of initFilter's unscaled and SWS_BILINEAR
 *    branches (utils.c) with the x86 filterAlign rules, for the vertical
 *    4:2:0 filter and for the horizontal filter an odd-width 4:2:x
 *    destination needs (swscale only reads chroma two-pixels-at-a-time
 *    when the width is even).
 *
 * The port is pinned to libswscale 10.1.102, from FFmpeg 9.0.2
 * (tools/ffmpeg.lock). Any FFmpeg bump must re-run
 * tests/test_cuda_encode_exact.py; selfTest() guards it at run time.
 *
 * Derived from libswscale (LGPL-2.1-or-later).
 */

#pragma once

#ifdef NELUX_ENABLE_CUDA

#include <Nelux/backends/cuda/RGBToYUVExact.cuh>
#include <cuda_runtime.h>

extern "C"
{
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace nelux::backends::cuda
{

namespace exact_detail
{

constexpr int kRgb2YuvShift = 15;

inline int64_t roundedDiv(int64_t a, int64_t b)
{
    return a >= 0 ? (a + (b >> 1)) / b : (a - (b >> 1)) / b;
}

inline int ilog2(unsigned v)
{
    int n = 0;
    while (v >>= 1)
        ++n;
    return n;
}

// Port of fill_rgb2yuv_table (libswscale/utils.c, 9.0.2).
inline void fillRgb2YuvTable(const int table[4], RgbToYuvExactParams& p)
{
    int64_t vr = table[0];
    int64_t ub = table[1];
    int64_t ug = -table[2];
    int64_t vg = -table[3];
    const int64_t ONE = 65536;
    int64_t cy = ONE;

    // swscale forces dstRange = 0 here; full range is a later pass.
    cy = cy * 255 / 219;

    const int64_t W = roundedDiv(ONE * ONE * ug, ub);
    const int64_t V = roundedDiv(ONE * ONE * vg, vr);
    const int64_t Z = ONE * ONE - W - V;

    const int64_t Cy = roundedDiv(cy * Z, ONE);
    const int64_t Cu = roundedDiv(ub * Z, ONE);
    const int64_t Cv = roundedDiv(vr * Z, ONE);

    const int64_t S = int64_t(1) << kRgb2YuvShift;
    p.ry = int32_t(-roundedDiv(S * V, Cy));
    p.gy = int32_t(roundedDiv(S * ONE * ONE, Cy));
    p.by = int32_t(-roundedDiv(S * W, Cy));

    p.ru = int32_t(roundedDiv(S * V, Cu));
    p.gu = int32_t(-roundedDiv(S * ONE * ONE, Cu));
    p.bu = int32_t(roundedDiv(S * (Z + W), Cu));

    p.rv = int32_t(roundedDiv(S * (V + Z), Cv));
    p.gv = int32_t(-roundedDiv(S * ONE * ONE, Cv));
    p.bv = int32_t(roundedDiv(S * W, Cv));

    // The table for SWS_CS_DEFAULT (ITU601 / SMPTE170M) is replaced by
    // rounded textbook constants.
    const int* def = sws_getCoefficients(SWS_CS_DEFAULT);
    if (!std::memcmp(table, def, sizeof(int) * 4))
    {
        p.by = int32_t(0.114 * 219 / 255 * S + 0.5);
        p.bv = -int32_t(0.081 * 224 / 255 * S + 0.5);
        p.bu = int32_t(0.500 * 224 / 255 * S + 0.5);
        p.gy = int32_t(0.587 * 219 / 255 * S + 0.5);
        p.gv = -int32_t(0.419 * 224 / 255 * S + 0.5);
        p.gu = -int32_t(0.331 * 224 / 255 * S + 0.5);
        p.ry = int32_t(0.299 * 219 / 255 * S + 0.5);
        p.rv = int32_t(0.500 * 224 / 255 * S + 0.5);
        p.ru = -int32_t(0.169 * 224 / 255 * S + 0.5);
    }
}

// Port of solve_range_convert + init_range_convert_constants for an RGB
// source (src_range is forced to 0) into a full-range destination of
// dstBpc <= 14, as the 15-bit lumRangeToJpeg / chrRangeToJpeg consume them.
inline void solveRangeConvert(uint16_t srcMin, uint16_t srcMax, uint16_t dstMin,
                              uint16_t dstMax, int srcShift, int multShift,
                              int32_t& coeff, int32_t& offset)
{
    const uint16_t srcRange = uint16_t(srcMax - srcMin);
    const uint16_t dstRange = uint16_t(dstMax - dstMin);
    const int totalShift = multShift + srcShift;
    const uint64_t q = (uint64_t(dstRange) << totalShift) / srcRange;
    const uint32_t c = uint32_t(-int64_t((-int64_t(q)) >> srcShift)); // AV_CEIL_RSHIFT
    const int64_t o = (int64_t(dstMax) << totalShift) -
                      (int64_t(srcMax) << srcShift) * int64_t(c) +
                      (int64_t(1) << (multShift - 1));
    coeff = int32_t(uint16_t(c)); // lumRangeToJpeg_c: uint16_t coeff = _coeff
    offset = int32_t(o);          // int32_t offset = _offset
}

inline void initRangeConvert(int dstBpc, RgbToYuvExactParams& p)
{
    const int bitDepth = dstBpc;
    const int srcShift = 15 - bitDepth;
    const int multShift = 14;
    const uint16_t mpegMin = uint16_t(16U << (bitDepth - 8));
    const uint16_t mpegMaxLum = uint16_t(235U << (bitDepth - 8));
    const uint16_t mpegMaxChr = uint16_t(240U << (bitDepth - 8));
    const uint16_t jpegMax = uint16_t((1U << bitDepth) - 1);
    solveRangeConvert(mpegMin, mpegMaxLum, 0, jpegMax, srcShift, multShift,
                      p.lumRangeCoeff, p.lumRangeOffset);
    solveRangeConvert(mpegMin, mpegMaxChr, 0, jpegMax, srcShift, multShift,
                      p.chrRangeCoeff, p.chrRangeOffset);
}

// Port of get_local_pos (utils.c) for the default (unset) chroma positions.
inline int localPos(int chrSubsample)
{
    int pos = (128 << chrSubsample) - 128;
    pos += 128;
    return pos >> chrSubsample;
}

struct Filter
{
    std::vector<int32_t> pos;
    std::vector<int16_t> coef;
    int taps = 0;
};

// Port of initFilter (utils.c, 9.0.2), restricted to what an SWS_BILINEAR
// legacy context takes: the unscaled branch and the generic branch with the
// bilinear kernel, no src/dst vectors, x86 (MMX) alignment rules.
inline bool initFilter(Filter& out, int xInc, int srcW, int dstW, int filterAlign,
                       int one, int srcPos, int dstPos)
{
    const int64_t fone = int64_t(1) << (54 - std::min(ilog2(unsigned(srcW / dstW)), 8));
    int filterSize;
    std::vector<int64_t> filter;
    std::vector<int32_t> filterPos(dstW + 3);

    if (std::abs(xInc - 0x10000) < 10 && srcPos == dstPos)
    {
        filterSize = 1;
        filter.assign(size_t(dstW) * filterSize, 0);
        for (int i = 0; i < dstW; i++)
        {
            filter[size_t(i) * filterSize] = fone;
            filterPos[i] = i;
        }
    }
    else
    {
        const int sizeFactor = 2; // SWS_BILINEAR
        if (xInc <= 1 << 16)
            filterSize = 1 + sizeFactor;
        else
            filterSize = 1 + (sizeFactor * srcW + dstW - 1) / dstW;
        filterSize = std::min(filterSize, srcW - 2);
        filterSize = std::max(filterSize, 1);

        filter.assign(size_t(dstW) * filterSize, 0);
        int64_t xDstInSrc = ((dstPos * int64_t(xInc)) >> 7) - ((srcPos * 0x10000LL) >> 7);
        for (int i = 0; i < dstW; i++)
        {
            int xx = int((xDstInSrc - (filterSize - 2) * (1LL << 16)) / (1 << 17));
            filterPos[i] = xx;
            for (int j = 0; j < filterSize; j++)
            {
                int64_t d = (std::llabs((int64_t(xx) * (1 << 17)) - xDstInSrc)) << 13;
                if (xInc > 1 << 16)
                    d = d * dstW / srcW;
                int64_t coeff = (1 << 30) - d;
                if (coeff < 0)
                    coeff = 0;
                coeff *= fone >> 30;
                filter[size_t(i) * filterSize + j] = coeff;
                xx++;
            }
            xDstInSrc += 2LL * xInc;
        }
    }

    // No src/dst filter vectors: filter2 == filter.
    const int filter2Size = filterSize;
    std::vector<int64_t> filter2 = filter;

    int minFilterSize = 0;
    for (int i = dstW - 1; i >= 0; i--)
    {
        int min = filter2Size;
        int64_t cutOff = 0;

        for (int j = 0; j < filter2Size; j++)
        {
            cutOff += std::llabs(filter2[size_t(i) * filter2Size]);
            if (double(cutOff) > 0.002 * double(fone))
                break;
            if (i < dstW - 1 && filterPos[i] >= filterPos[i + 1])
                break;
            int k;
            for (k = 1; k < filter2Size; k++)
                filter2[size_t(i) * filter2Size + k - 1] = filter2[size_t(i) * filter2Size + k];
            filter2[size_t(i) * filter2Size + k - 1] = 0;
            filterPos[i]++;
        }

        cutOff = 0;
        for (int j = filter2Size - 1; j > 0; j--)
        {
            cutOff += std::llabs(filter2[size_t(i) * filter2Size + j]);
            if (double(cutOff) > 0.002 * double(fone))
                break;
            min--;
        }

        if (min > minFilterSize)
            minFilterSize = min;
    }

    // x86: special case for unscaled vertical filtering.
    if (minFilterSize == 1 && filterAlign == 2)
        filterAlign = 1;

    filterSize = (minFilterSize + (filterAlign - 1)) & (~(filterAlign - 1));
    filter.assign(size_t(dstW) * filterSize, 0);
    for (int i = 0; i < dstW; i++)
        for (int j = 0; j < filterSize; j++)
            filter[size_t(i) * filterSize + j] =
                j >= filter2Size ? 0 : filter2[size_t(i) * filter2Size + j];

    // Fix borders.
    for (int i = 0; i < dstW; i++)
    {
        int64_t* f = &filter[size_t(i) * filterSize];
        if (filterPos[i] < 0)
        {
            for (int j = 1; j < filterSize; j++)
            {
                const int left = std::max(j + filterPos[i], 0);
                f[left] += f[j];
                f[j] = 0;
            }
            filterPos[i] = 0;
        }

        if (filterPos[i] + filterSize > srcW)
        {
            const int shift = filterPos[i] + std::min(filterSize - srcW, 0);
            int64_t acc = 0;
            for (int j = filterSize - 1; j >= 0; j--)
            {
                if (filterPos[i] + j >= srcW)
                {
                    acc += f[j];
                    f[j] = 0;
                }
            }
            for (int j = filterSize - 1; j >= 0; j--)
                f[j] = j < shift ? 0 : f[j - shift];
            filterPos[i] -= shift;
            f[srcW - 1 - filterPos[i]] += acc;
        }
        if (filterPos[i] < 0 || filterPos[i] >= srcW)
            return false;
    }

    // Normalize with error diffusion.
    out.taps = filterSize;
    out.pos.assign(filterPos.begin(), filterPos.begin() + dstW);
    out.coef.assign(size_t(dstW) * filterSize, 0);
    for (int i = 0; i < dstW; i++)
    {
        int64_t error = 0;
        int64_t sum = 0;
        for (int j = 0; j < filterSize; j++)
            sum += filter[size_t(i) * filterSize + j];
        sum = (sum + one / 2) / one;
        if (!sum)
            sum = 1;
        for (int j = 0; j < filterSize; j++)
        {
            const int64_t v = filter[size_t(i) * filterSize + j] + error;
            const int intV = int(roundedDiv(v, sum));
            out.coef[size_t(i) * filterSize + j] = int16_t(intV);
            error = v - intV * sum;
        }
    }
    return true;
}

} // namespace exact_detail

// Host-side plan: everything the kernels need besides the pixels. Built once
// per encoder; the filter tables are uploaded once by the converter.
struct RgbToYuvExactPlan
{
    int width = 0;
    int height = 0;
    AVPixelFormat srcFmt = AV_PIX_FMT_NONE;
    AVPixelFormat dstFmt = AV_PIX_FMT_NONE;
    ExactRgbSource source = ExactRgbSource::Rgb24;
    ExactYuvTarget target = ExactYuvTarget::Yuv420p;
    RgbToYuvExactParams params{};
    exact_detail::Filter vChr; // 4:2:0 only
    exact_detail::Filter hChr; // odd-width 4:2:x only

    int chromaShiftX() const
    {
        return (target == ExactYuvTarget::Yuv444p || target == ExactYuvTarget::Yuv444p10) ? 0 : 1;
    }
    int chromaShiftY() const
    {
        return (target == ExactYuvTarget::Yuv420p || target == ExactYuvTarget::Nv12 ||
                target == ExactYuvTarget::Yuv420p10 || target == ExactYuvTarget::P010)
                   ? 1
                   : 0;
    }
    int bitDepth() const
    {
        return (target == ExactYuvTarget::Yuv420p10 || target == ExactYuvTarget::Yuv422p10 ||
                target == ExactYuvTarget::Yuv444p10 || target == ExactYuvTarget::P010)
                   ? 10
                   : 8;
    }
    int chromaWidth() const { return (width + chromaShiftX()) >> chromaShiftX(); }
    int chromaHeight() const { return (height + chromaShiftY()) >> chromaShiftY(); }
};

// Same AVColorSpace -> SWS_CS mapping (and HD/SD default) as
// RGBToAutoConverter.
inline int exactSwsColorspace(AVColorSpace cs, int height)
{
    if (cs == AVCOL_SPC_UNSPECIFIED)
        cs = (height > 576) ? AVCOL_SPC_BT709 : AVCOL_SPC_BT470BG;
    switch (cs)
    {
    case AVCOL_SPC_BT709:      return SWS_CS_ITU709;
    case AVCOL_SPC_FCC:        return SWS_CS_FCC;
    case AVCOL_SPC_BT470BG:    return SWS_CS_ITU601;
    case AVCOL_SPC_SMPTE170M:  return SWS_CS_SMPTE170M;
    case AVCOL_SPC_SMPTE240M:  return SWS_CS_SMPTE240M;
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:  return SWS_CS_BT2020;
    default:                   return SWS_CS_ITU601;
    }
}

// Same range rule as RGBToAutoConverter: explicit > YUVJ pixfmt > limited.
inline bool exactFullRange(AVPixelFormat dst, AVColorRange range)
{
    if (range == AVCOL_RANGE_JPEG)
        return true;
    if (range == AVCOL_RANGE_MPEG)
        return false;
    return dst == AV_PIX_FMT_YUVJ420P || dst == AV_PIX_FMT_YUVJ422P ||
           dst == AV_PIX_FMT_YUVJ444P;
}

inline bool exactSourceFor(AVPixelFormat src, ExactRgbSource& out)
{
    switch (src)
    {
    case AV_PIX_FMT_RGB24:    out = ExactRgbSource::Rgb24;  return true;
    case AV_PIX_FMT_RGBA:     out = ExactRgbSource::Rgba;   return true;
    case AV_PIX_FMT_RGB48LE:  out = ExactRgbSource::Rgb48;  return true;
    case AV_PIX_FMT_RGBA64LE: out = ExactRgbSource::Rgba64; return true;
    default:                  return false;
    }
}

inline bool exactTargetFor(AVPixelFormat dst, ExactYuvTarget& out)
{
    switch (dst)
    {
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:    out = ExactYuvTarget::Yuv420p;   return true;
    case AV_PIX_FMT_NV12:        out = ExactYuvTarget::Nv12;      return true;
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:    out = ExactYuvTarget::Yuv422p;   return true;
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:    out = ExactYuvTarget::Yuv444p;   return true;
    case AV_PIX_FMT_YUV420P10LE: out = ExactYuvTarget::Yuv420p10; return true;
    case AV_PIX_FMT_YUV422P10LE: out = ExactYuvTarget::Yuv422p10; return true;
    case AV_PIX_FMT_YUV444P10LE: out = ExactYuvTarget::Yuv444p10; return true;
    case AV_PIX_FMT_P010LE:      out = ExactYuvTarget::P010;      return true;
    default:                     return false;
    }
}

// Builds the plan for a same-size SWS_BILINEAR conversion. Returns false
// when the pair is not covered (the caller keeps the CPU swscale path).
inline bool buildRgbToYuvExactPlan(int width, int height, AVPixelFormat src,
                                   AVPixelFormat dst, AVColorSpace colorspace,
                                   AVColorRange colorRange, RgbToYuvExactPlan& plan)
{
    using namespace exact_detail;
    // The kernels follow swscale's x86 code paths. An aarch64 or plain-C
    // libswscale writes other bytes in the 4:2:0 chroma rows (by 1 LSB), so
    // the plan is still a correct conversion there, just not the CPU path's
    // bytes: callers that need those bytes run selfTest() first.
    if (width <= 0 || height <= 0)
        return false;
    if (!exactSourceFor(src, plan.source) || !exactTargetFor(dst, plan.target))
        return false;

    // A 16-bit source into an 8-bit destination is the one pair where
    // swscale dithers (ff_dither_8x8_128; should_dither is set for 16-bit
    // sources). nelux never stages 16-bit RGB for an 8-bit destination, so
    // it is left to the CPU path.
    const bool deepSource = plan.source == ExactRgbSource::Rgb48 ||
                            plan.source == ExactRgbSource::Rgba64;
    if (deepSource && plan.bitDepth() == 8)
        return false;

    plan.width = width;
    plan.height = height;
    plan.srcFmt = src;
    plan.dstFmt = dst;
    plan.params = RgbToYuvExactParams{};
    plan.vChr = Filter{};
    plan.hChr = Filter{};

    fillRgb2YuvTable(sws_getCoefficients(exactSwsColorspace(colorspace, height)), plan.params);
    plan.params.rangeConvert = exactFullRange(dst, colorRange) ? 1 : 0;
    if (plan.params.rangeConvert)
        initRangeConvert(plan.bitDepth(), plan.params);

    const int sx = plan.chromaShiftX();
    const int sy = plan.chromaShiftY();
    const int chrDstW = plan.chromaWidth();
    const int chrDstH = plan.chromaHeight();

    // swscale averages pixel pairs on input only for an even width; an odd
    // width keeps full-resolution chroma and scales it horizontally.
    if (sx && (width & 1))
    {
        const int chrSrcW = width;
        const int xInc = int(((int64_t(chrSrcW) << 16) + (chrDstW >> 1)) / chrDstW);
        if (!initFilter(plan.hChr, xInc, chrSrcW, chrDstW, 4, 1 << 14, localPos(0), localPos(1)))
            return false;
        if (plan.hChr.taps > kExactMaxHTaps)
            return false;
    }

    if (sy)
    {
        const int yInc = int(((int64_t(height) << 16) + (chrDstH >> 1)) / chrDstH);
        if (!initFilter(plan.vChr, yInc, height, chrDstH, 2, 1 << 12, localPos(0), localPos(1)))
            return false;
        if (plan.vChr.taps > kExactMaxVTaps)
            return false;
    }
    return true;
}

/**
 * @brief Device-resident exact converter for one (size, source, target,
 *        colorspace, range). Uploads the filter tables once at construction
 *        and waits for them; convert() only enqueues kernels.
 *
 * Usage (CUDA tensor -> software encoder):
 *   if (RgbToYuvExactConverter::supported(w, h, src, dst, cs, range))
 *       conv = std::make_unique<RgbToYuvExactConverter>(w, h, src, dst, cs, range, stream);
 *   if (conv->selfTest(cpuConvert, stream) != SelfTestResult::Match)
 *       conv.reset();  // once, keep CPU path
 *   conv->convert(rgb, dstPlanes, torchStream);   // then D2H the planes
 */
class RgbToYuvExactConverter
{
  public:
    // Converts RGB on the CPU the way the encoder's CPU path does (nelux:
    // RGBToAutoConverter::convert into @p out). The buffer is packed, with
    // the plan's source format.
    using CpuConvertFn = std::function<void(const void* rgb, AVFrame* out)>;

    // Match / Mismatch: the comparison ran. Failed: it could not (a host or
    // device allocation, a copy, the reference convert or the launch failed),
    // which proves nothing about the kernels.
    enum class SelfTestResult { Match, Mismatch, Failed };

    static bool supported(int width, int height, AVPixelFormat src, AVPixelFormat dst,
                          AVColorSpace colorspace, AVColorRange colorRange)
    {
        RgbToYuvExactPlan plan;
        return buildRgbToYuvExactPlan(width, height, src, dst, colorspace, colorRange, plan);
    }

    // @p uploadStream carries the one-time table upload, which is waited on
    // here (a pageable cudaMemcpy may return before its DMA lands, and a
    // later launch on a non-blocking stream would not be ordered after it).
    RgbToYuvExactConverter(int width, int height, AVPixelFormat src, AVPixelFormat dst,
                           AVColorSpace colorspace, AVColorRange colorRange,
                           cudaStream_t uploadStream = nullptr)
    {
        if (!buildRgbToYuvExactPlan(width, height, src, dst, colorspace, colorRange, plan_))
            throw std::runtime_error("RgbToYuvExactConverter: unsupported conversion");
        try
        {
            upload(plan_.vChr, vPos_, vCoef_, uploadStream);
            upload(plan_.hChr, hPos_, hCoef_, uploadStream);
            if (cudaStreamSynchronize(uploadStream) != cudaSuccess)
                throw std::runtime_error("RgbToYuvExactConverter: table upload failed");
        }
        catch (...)
        {
            release();
            throw;
        }
        plan_.params.vChrPos = vPos_;
        plan_.params.vChrCoef = vCoef_;
        plan_.params.vChrTaps = plan_.vChr.taps;
        plan_.params.hChrPos = hPos_;
        plan_.params.hChrCoef = hCoef_;
        plan_.params.hChrTaps = plan_.hChr.taps;
    }

    ~RgbToYuvExactConverter() { release(); }

    RgbToYuvExactConverter(const RgbToYuvExactConverter&) = delete;
    RgbToYuvExactConverter& operator=(const RgbToYuvExactConverter&) = delete;

    const RgbToYuvExactPlan& plan() const { return plan_; }

    // Whether convert() takes @p src. Integer elements: interleaved pixels
    // of exactly the source's channels, or planar channels. Float elements:
    // any strides, where the encoder stages a float frame (8-bit samples
    // into an 8-bit target, 16-bit samples into a 10-bit one), except an
    // odd-width 4:2:x target.
    bool reads(const RgbToYuvExactSource& src) const
    {
        const bool alpha = plan_.source == ExactRgbSource::Rgba ||
                           plan_.source == ExactRgbSource::Rgba64;
        const bool deepSource = plan_.source == ExactRgbSource::Rgb48 ||
                                plan_.source == ExactRgbSource::Rgba64;
        if (!src.data)
            return false;
        if (src.element != ExactElement::Integer)
            return deepSource == (plan_.bitDepth() > 8) &&
                   !(plan_.chromaShiftX() && (plan_.width & 1));
        return src.pixelStride == 1 ||
               (src.channelStride == 1 && src.pixelStride == (alpha ? 4 : 3));
    }

    // Enqueue on @p stream; the caller owns ordering (event / sync).
    void convert(const RgbToYuvExactSource& src, const RgbToYuvExactDest& dst,
                 cudaStream_t stream) const
    {
        const cudaError_t err = RgbToYuvExact(plan_.params, plan_.source, plan_.target,
                                              plan_.width, plan_.height, src, dst, stream);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("RgbToYuvExactConverter: launch failed: ") +
                                     cudaGetErrorString(err));
    }

    /**
     * @brief Convert one synthetic frame (random samples mixed with 0/max
     *        extremes) at the plan's real size both ways and compare every
     *        byte. Run once before trusting the GPU path: it catches a
     *        libswscale that no longer behaves like 9.0.2 (an FFmpeg bump or a
     *        patched build) and keeps the encoder on its CPU path. Blocking;
     *        allocates one frame's worth of host and device memory and runs
     *        up to 8 short-lived threads (the fill, and the CPU convert
     *        beside the GPU one) for the duration of the call.
     */
    SelfTestResult selfTest(const CpuConvertFn& cpuConvert, cudaStream_t stream) const
    {
        const AVPixFmtDescriptor* sd = av_pix_fmt_desc_get(plan_.srcFmt);
        const int channels = sd->nb_components;
        const int elemBytes = sd->comp[0].depth > 8 ? 2 : 1;
        const int w = plan_.width, h = plan_.height;
        const size_t rowBytes = size_t(w) * channels * elemBytes;

        // 8 bytes per step from two xorshift64 streams: `a` gives the samples,
        // 3 bits of `b` per sample pick 0 (1 in 8), max (1 in 8) or keep it,
        // as lane masks so no per-sample branch (the per-sample loop cost
        // ~50 ms at 4K). rowBytes is a whole number of samples. The steps are
        // serial (~8 ms of a 4K frame on one core), so a large frame is
        // filled in up to 8 slices of 1 MiB or more on threads of their own
        // (~4 ms), each with its own seeds.
        const size_t bytes = rowBytes * h;
        std::unique_ptr<uint8_t[]> rgb(new uint8_t[bytes]);
        const int laneBits = 8 * elemBytes;
        const uint64_t lo = elemBytes == 1 ? 0x0101010101010101ull : 0x0001000100010001ull;
        const uint64_t hi = lo << (laneBits - 1);
        const uint64_t laneMax = elemBytes == 1 ? 0xFFu : 0xFFFFu;
        const size_t slices = std::clamp<size_t>(bytes >> 20, 1, 8);
        const size_t sliceBytes = (bytes / slices + 7) & ~size_t(7);
        auto fill = [&](size_t slice) {
            uint64_t a = 0x9E3779B97F4A7C15ull * (2 * slice + 1); // odd, so never 0
            uint64_t b = 0xD1B54A32D192ED03ull * (2 * slice + 1);
            const size_t end = std::min(bytes, (slice + 1) * sliceBytes);
            for (size_t i = slice * sliceBytes; i < end; i += 8)
            {
                a ^= a << 13; a ^= a >> 7; a ^= a << 17; // xorshift64
                b ^= b << 13; b ^= b >> 7; b ^= b << 17;
                const uint64_t mode = b & (lo * 7); // 0..7 per lane, no carry out
                const uint64_t zero = (hi & ~((mode + hi - lo) & hi)) >> (laneBits - 1);
                const uint64_t full = (hi & ~(((mode ^ lo) + hi - lo) & hi)) >> (laneBits - 1);
                const uint64_t v = (a & ~((zero | full) * laneMax)) | full * laneMax;
                std::memcpy(rgb.get() + i, &v, std::min<size_t>(8, end - i));
            }
        };
        {
            std::vector<std::thread> fillers;
            fillers.reserve(slices);
            size_t next = 1;
            try
            {
                for (; next < slices; ++next)
                    fillers.emplace_back(fill, next);
            }
            catch (const std::system_error&)
            {
                // No thread to spare: this one fills the rest.
            }
            for (size_t slice = next; slice < slices; ++slice)
                fill(slice);
            fill(0);
            for (auto& t : fillers)
                t.join();
        }

        AVFrame* ref = av_frame_alloc();
        if (!ref)
            return SelfTestResult::Failed;
        ref->format = plan_.dstFmt;
        ref->width = w;
        ref->height = h;

        // The reference convert (sws_scale, ~13 ms at 4K) runs on a thread of
        // its own while this one uploads, converts and downloads (~5 ms).
        bool refOk = av_frame_get_buffer(ref, 32) >= 0;
        auto reference = [&] {
            try
            {
                if (refOk)
                    cpuConvert(rgb.get(), ref);
            }
            catch (...)
            {
                refOk = false; // no reference, no proof
            }
        };
        std::thread refThread;
        try
        {
            refThread = std::thread(reference);
        }
        catch (const std::system_error&)
        {
            reference();
        }
        // Joined on every way out: the thread reads rgb and writes ref.
        struct Joiner
        {
            std::thread& t;
            ~Joiner()
            {
                if (t.joinable())
                    t.join();
            }
        } joiner{refThread};

        const int planes = av_pix_fmt_count_planes(plan_.dstFmt);
        const AVPixFmtDescriptor* dd = av_pix_fmt_desc_get(plan_.dstFmt);
        void* dSrc = nullptr;
        void* dDst[3] = {nullptr, nullptr, nullptr};
        RgbToYuvExactDest dst{};
        size_t planeBytes[3] = {0, 0, 0};
        int planeRows[3] = {0, 0, 0};
        bool ok = cudaMalloc(&dSrc, bytes) == cudaSuccess &&
                  cudaMemcpyAsync(dSrc, rgb.get(), bytes, cudaMemcpyHostToDevice, stream) ==
                      cudaSuccess;
        for (int p = 0; ok && p < planes; p++)
        {
            planeRows[p] = p == 0 ? h : AV_CEIL_RSHIFT(h, dd->log2_chroma_h);
            planeBytes[p] = size_t(av_image_get_linesize(plan_.dstFmt, w, p));
            ok = cudaMalloc(&dDst[p], planeBytes[p] * planeRows[p]) == cudaSuccess;
            dst.plane[p] = dDst[p];
            dst.pitch[p] = int64_t(planeBytes[p]);
        }
        if (ok)
        {
            const RgbToYuvExactSource src{dSrc, int64_t(w) * channels, channels, 1};
            ok = RgbToYuvExact(plan_.params, plan_.source, plan_.target, w, h, src, dst,
                               stream) == cudaSuccess &&
                 cudaStreamSynchronize(stream) == cudaSuccess;
        }
        std::vector<uint8_t> got[3];
        for (int p = 0; ok && p < planes; p++)
        {
            got[p].resize(planeBytes[p] * planeRows[p]);
            ok = cudaMemcpy(got[p].data(), dDst[p], got[p].size(), cudaMemcpyDeviceToHost) ==
                 cudaSuccess;
        }

        if (refThread.joinable())
            refThread.join();
        ok = ok && refOk;
        bool same = true;
        for (int p = 0; ok && same && p < planes; p++)
            for (int y = 0; same && y < planeRows[p]; y++)
                same = std::memcmp(got[p].data() + planeBytes[p] * y,
                                   ref->data[p] + int64_t(ref->linesize[p]) * y, planeBytes[p]) == 0;

        cudaFree(dSrc);
        for (void* d : dDst)
            cudaFree(d);
        av_frame_free(&ref);
        if (!ok)
        {
            cudaGetLastError(); // a failed malloc or launch must not fail the next call
            return SelfTestResult::Failed;
        }
        return same ? SelfTestResult::Match : SelfTestResult::Mismatch;
    }

  private:
    template <typename T>
    static T* uploadArray(const std::vector<T>& v, cudaStream_t stream)
    {
        if (v.empty())
            return nullptr;
        T* d = nullptr;
        if (cudaMalloc(&d, v.size() * sizeof(T)) != cudaSuccess)
            throw std::runtime_error("RgbToYuvExactConverter: cudaMalloc failed");
        if (cudaMemcpyAsync(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice, stream) !=
            cudaSuccess)
        {
            cudaFree(d);
            throw std::runtime_error("RgbToYuvExactConverter: table upload failed");
        }
        return d;
    }

    static void upload(const exact_detail::Filter& f, int32_t*& pos, int16_t*& coef,
                       cudaStream_t stream)
    {
        pos = uploadArray(f.pos, stream);
        coef = uploadArray(f.coef, stream);
    }

    void release()
    {
        cudaFree(vPos_);
        cudaFree(vCoef_);
        cudaFree(hPos_);
        cudaFree(hCoef_);
        vPos_ = hPos_ = nullptr;
        vCoef_ = hCoef_ = nullptr;
    }

    RgbToYuvExactPlan plan_;
    int32_t* vPos_ = nullptr;
    int16_t* vCoef_ = nullptr;
    int32_t* hPos_ = nullptr;
    int16_t* hCoef_ = nullptr;
};

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
