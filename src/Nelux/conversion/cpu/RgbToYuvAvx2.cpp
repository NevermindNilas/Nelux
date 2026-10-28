#include "RgbToYuvAvx2.hpp"

extern "C"
{
#include <libavutil/cpu.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

// x86-64 only: elsewhere (macOS arm64) swscale runs other SIMD with other
// rounding, so the table is empty and every conversion stays on swscale.
#if defined(__x86_64__) || defined(_M_X64)
#define NELUX_RGB2YUV_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define NELUX_RGB2YUV_X86 0
#endif

// The build does not enable AVX2 globally (see build-flag notes in
// CMakeLists.txt), so every function that touches 256-bit intrinsics carries
// the target attribute on GCC/Clang. MSVC accepts the intrinsics in any
// translation unit and needs no attribute.
#if defined(_MSC_VER) && !defined(__clang__)
#define NELUX_AVX2
#else
#define NELUX_AVX2 __attribute__((target("avx2")))
#endif

namespace nelux
{
namespace conversion
{
namespace cpu
{
namespace simd
{

bool cpuHasAvx2()
{
#if !NELUX_RGB2YUV_X86
    return false;
#else
    static const bool hasAvx2 = []
    {
#if defined(_MSC_VER)
        int regs[4];
        __cpuid(regs, 0);
        if (regs[0] < 7)
            return false;
        __cpuid(regs, 1);
        const bool osxsave = (regs[2] & (1 << 27)) != 0;
        const bool avx = (regs[2] & (1 << 28)) != 0;
        if (!osxsave || !avx || (_xgetbv(0) & 0x6) != 0x6)
            return false;
        __cpuidex(regs, 7, 0);
        return (regs[1] & (1 << 5)) != 0;
#else
        unsigned a, b, c, d;
        if (!__get_cpuid(0, &a, &b, &c, &d) || a < 7)
            return false;
        __get_cpuid(1, &a, &b, &c, &d);
        const bool osxsave = (c & (1u << 27)) != 0;
        const bool avx = (c & (1u << 28)) != 0;
        if (!osxsave || !avx)
            return false;
        unsigned xcr0Lo, xcr0Hi;
        __asm__ volatile("xgetbv" : "=a"(xcr0Lo), "=d"(xcr0Hi) : "c"(0));
        if ((xcr0Lo & 0x6) != 0x6)
            return false;
        __get_cpuid_count(7, 0, &a, &b, &c, &d);
        return (b & (1u << 5)) != 0;
#endif
    }();
    return hasAvx2;
#endif
}

bool swscaleIsVerifiedVersion()
{
    return swscale_version() == AV_VERSION_INT(10, 1, 102);
}

namespace
{

#if NELUX_RGB2YUV_X86
// ---------------------------------------------------------------------------
// Plan construction: ports of the libswscale 10.1.102 init code nelux depends on.
// Plans exist only on x86: elsewhere swscale runs other SIMD paths (and
// av_get_cpu_flags() bits mean other things), so makePlan() is a stub.
// ---------------------------------------------------------------------------

int64_t roundedDiv(int64_t a, int64_t b)
{
    return (a >= 0 ? a + (b >> 1) : a - (b >> 1)) / b;
}

// fill_rgb2yuv_table (utils.c). dstRange is forced to 0 there ("range = 1 is
// handled elsewhere"), and a table equal to SWS_CS_DEFAULT's (ITU601,
// SMPTE170M) gets hard-coded BT.601 values instead of the derived ones.
void fillRgb2YuvTable(const int table[4], RgbToYuvPlan& p)
{
    int64_t vr = table[0];
    int64_t ub = table[1];
    int64_t ug = -table[2];
    int64_t vg = -table[3];
    const int64_t one = 65536;
    const int64_t cy = one * 255 / 219;
    const int64_t w = roundedDiv(one * one * ug, ub);
    const int64_t v = roundedDiv(one * one * vg, vr);
    const int64_t z = one * one - w - v;
    const int64_t cY = roundedDiv(cy * z, one);
    const int64_t cU = roundedDiv(ub * z, one);
    const int64_t cV = roundedDiv(vr * z, one);
    const int64_t shift = int64_t(1) << 15;

    p.ry = int32_t(-roundedDiv(shift * v, cY));
    p.gy = int32_t(roundedDiv(shift * one * one, cY));
    p.by = int32_t(-roundedDiv(shift * w, cY));
    p.ru = int32_t(roundedDiv(shift * v, cU));
    p.gu = int32_t(-roundedDiv(shift * one * one, cU));
    p.bu = int32_t(roundedDiv(shift * (z + w), cU));
    p.rv = int32_t(roundedDiv(shift * (v + z), cV));
    p.gv = int32_t(-roundedDiv(shift * one * one, cV));
    p.bv = int32_t(roundedDiv(shift * w, cV));

    if (!std::memcmp(table, sws_getCoefficients(SWS_CS_DEFAULT), 4 * sizeof(int)))
    {
        p.by = ((int)(0.114 * 219 / 255 * (1 << 15) + 0.5));
        p.bv = (-(int)(0.081 * 224 / 255 * (1 << 15) + 0.5));
        p.bu = ((int)(0.500 * 224 / 255 * (1 << 15) + 0.5));
        p.gy = ((int)(0.587 * 219 / 255 * (1 << 15) + 0.5));
        p.gv = (-(int)(0.419 * 224 / 255 * (1 << 15) + 0.5));
        p.gu = (-(int)(0.331 * 224 / 255 * (1 << 15) + 0.5));
        p.ry = ((int)(0.299 * 219 / 255 * (1 << 15) + 0.5));
        p.rv = ((int)(0.500 * 224 / 255 * (1 << 15) + 0.5));
        p.ru = (-(int)(0.169 * 224 / 255 * (1 << 15) + 0.5));
    }
}

// solve_range_convert / init_range_convert_constants (swscale.c), limited
// source (RGB input always is) to full-range destination, dstBpc <= 14.
void solveRangeConvert(int bitDepth, uint16_t srcMax, uint32_t& coeff, int64_t& offset)
{
    const int srcShift = 15 - bitDepth;
    const int multShift = 14;
    const uint16_t srcMin = uint16_t(16u << (bitDepth - 8));
    const uint16_t dstMax = uint16_t((1u << bitDepth) - 1);
    const uint16_t srcRange = uint16_t(srcMax - srcMin);
    const uint16_t dstRange = dstMax;
    const int totalShift = multShift + srcShift;
    const uint64_t q = (uint64_t(dstRange) << totalShift) / srcRange;
    coeff = uint32_t((q + (uint64_t(1) << srcShift) - 1) >> srcShift);
    offset = (int64_t(dstMax) << totalShift) - (int64_t(srcMax) << srcShift) * coeff +
             (1u << (multShift - 1));
}

int log2Floor(int v)
{
    int n = 0;
    while (v > 1)
    {
        v >>= 1;
        ++n;
    }
    return n;
}

// initFilter (utils.c) for SWS_BILINEAR with no user filters on x86. Only the
// branches a same-size RGB->YUV context reaches are kept.
bool initFilterBilinear(int xInc, int srcW, int dstW, int filterAlign, int one,
                        int srcPos, int dstPos, bool cpuMmx, std::vector<int16_t>& outFilter,
                        std::vector<int32_t>& outPos, int& outSize)
{
    const int64_t fone = int64_t(1) << (54 - std::min(log2Floor(srcW / dstW), 8));
    std::vector<int32_t> pos(dstW);
    std::vector<int64_t> filter;
    int filterSize;

    if (std::abs(xInc - 0x10000) < 10 && srcPos == dstPos)
    {
        filterSize = 1;
        filter.assign(dstW, 0);
        for (int i = 0; i < dstW; i++)
        {
            filter[i] = fone;
            pos[i] = i;
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
        int64_t xDstInSrc = ((dstPos * (int64_t)xInc) >> 7) - ((srcPos * 0x10000LL) >> 7);
        for (int i = 0; i < dstW; i++)
        {
            int xx = int((xDstInSrc - (filterSize - 2) * (1LL << 16)) / (1 << 17));
            pos[i] = xx;
            for (int j = 0; j < filterSize; j++)
            {
                int64_t d = std::llabs(((int64_t)xx * (1 << 17)) - xDstInSrc) << 13;
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

    // No source/destination vectors: filter2 is the filter itself.
    const int filter2Size = filterSize;
    std::vector<int64_t>& filter2 = filter;

    int minFilterSize = 0;
    for (int i = dstW - 1; i >= 0; i--)
    {
        int minSize = filter2Size;
        int64_t cutOff = 0;
        for (int j = 0; j < filter2Size; j++)
        {
            cutOff += std::llabs(filter2[size_t(i) * filter2Size]);
            if (cutOff > SWS_MAX_REDUCE_CUTOFF * fone)
                break;
            if (i < dstW - 1 && pos[i] >= pos[i + 1])
                break;
            int k;
            for (k = 1; k < filter2Size; k++)
                filter2[size_t(i) * filter2Size + k - 1] = filter2[size_t(i) * filter2Size + k];
            filter2[size_t(i) * filter2Size + k - 1] = 0;
            pos[i]++;
        }
        cutOff = 0;
        for (int j = filter2Size - 1; j > 0; j--)
        {
            cutOff += std::llabs(filter2[size_t(i) * filter2Size + j]);
            if (cutOff > SWS_MAX_REDUCE_CUTOFF * fone)
                break;
            minSize--;
        }
        if (minSize > minFilterSize)
            minFilterSize = minSize;
    }

    // Unscaled vertical special case (HAVE_MMX && AV_CPU_FLAG_MMX).
    if (cpuMmx && minFilterSize == 1 && filterAlign == 2)
        filterAlign = 1;
    if (minFilterSize <= 0)
        return false;

    filterSize = (minFilterSize + (filterAlign - 1)) & ~(filterAlign - 1);
    std::vector<int64_t> f(size_t(dstW) * filterSize, 0);
    for (int i = 0; i < dstW; i++)
        for (int j = 0; j < filterSize; j++)
            f[size_t(i) * filterSize + j] =
                j >= filter2Size ? 0 : filter2[size_t(i) * filter2Size + j];

    // Fix borders.
    for (int i = 0; i < dstW; i++)
    {
        int64_t* row = &f[size_t(i) * filterSize];
        if (pos[i] < 0)
        {
            for (int j = 1; j < filterSize; j++)
            {
                const int left = std::max(j + pos[i], 0);
                row[left] += row[j];
                row[j] = 0;
            }
            pos[i] = 0;
        }
        if (pos[i] + filterSize > srcW)
        {
            const int shift = pos[i] + std::min(filterSize - srcW, 0);
            int64_t acc = 0;
            for (int j = filterSize - 1; j >= 0; j--)
            {
                if (pos[i] + j >= srcW)
                {
                    acc += row[j];
                    row[j] = 0;
                }
            }
            for (int j = filterSize - 1; j >= 0; j--)
            {
                if (j < shift)
                    row[j] = 0;
                else
                    row[j] = row[j - shift];
            }
            pos[i] -= shift;
            row[srcW - 1 - pos[i]] += acc;
        }
        if (pos[i] < 0 || pos[i] >= srcW)
            return false;
    }

    // Normalize with error diffusion.
    outFilter.assign(size_t(dstW) * filterSize, 0);
    for (int i = 0; i < dstW; i++)
    {
        int64_t error = 0;
        int64_t sum = 0;
        for (int j = 0; j < filterSize; j++)
            sum += f[size_t(i) * filterSize + j];
        sum = (sum + one / 2) / one;
        if (!sum)
            sum = 1;
        for (int j = 0; j < filterSize; j++)
        {
            const int64_t v = f[size_t(i) * filterSize + j] + error;
            const int intV = int(roundedDiv(v, sum));
            outFilter[size_t(i) * filterSize + j] = int16_t(intV);
            error = v - intV * sum;
        }
    }
    outPos = pos;
    outSize = filterSize;
    return true;
}
#endif // NELUX_RGB2YUV_X86

// ---------------------------------------------------------------------------
// Scalar model. Each helper is one swscale stage on one sample; the AVX2
// kernel uses the same helpers for row tails.
// ---------------------------------------------------------------------------

inline int clampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

inline unsigned readSample(const uint8_t* row, int x, int channel, const RgbToYuvPlan& p)
{
    const uint8_t* px = row + size_t(x) * p.srcPixelBytes;
    if (!p.deepSource)
        return px[channel];
    return unsigned(px[2 * channel]) | (unsigned(px[2 * channel + 1]) << 8);
}

// RangeToJpeg as the SIMD converter applies it: pmaddwd with the low word of
// the coefficient, low dword of the offset, psrad 14, packssdw.
inline int rangeToJpeg(int s, int32_t coeff, int32_t offset)
{
    const int64_t v = int64_t(s) * int16_t(coeff & 0xFFFF) + offset;
    return clampInt(int32_t(uint32_t(v)) >> 14, -32768, 32767);
}

// Luma after lumToYV12 + identity hscale (+ range): the 15-bit intermediate.
inline int lumaS15(const RgbToYuvPlan& p, unsigned r, unsigned g, unsigned b)
{
    int s;
    if (!p.deepSource)
    {
        const int y14 = (p.ry * int(r) + p.gy * int(g) + p.by * int(b) + 0x80100) >> 9;
        s = clampInt(2 * int16_t(y14), -32768, 32767);
    }
    else
    {
        const uint16_t y16 = uint16_t((unsigned(p.ry) * r + unsigned(p.gy) * g +
                                       unsigned(p.by) * b + 0x8004000u) >> 15);
        s = y16 >> 1;
    }
    return p.fullRange ? rangeToJpeg(s, p.lumRangeCoeff, p.lumRangeOffset) : s;
}

// chrToYV12 on one full-resolution pixel: 14-bit (8-bit source) or 16-bit
// unsigned (deep source) chroma, before hcScale.
inline void chromaRaw(const RgbToYuvPlan& p, unsigned r, unsigned g, unsigned b, int& u, int& v)
{
    if (!p.deepSource)
    {
        u = int16_t((p.ru * int(r) + p.gu * int(g) + p.bu * int(b) + 0x400100) >> 9);
        v = int16_t((p.rv * int(r) + p.gv * int(g) + p.bv * int(b) + 0x400100) >> 9);
    }
    else
    {
        u = uint16_t((unsigned(p.ru) * r + unsigned(p.gu) * g + unsigned(p.bu) * b +
                      0x40004000u) >> 15);
        v = uint16_t((unsigned(p.rv) * r + unsigned(p.gv) * g + unsigned(p.bv) * b +
                      0x40004000u) >> 15);
    }
}

// Identity hcScale (hscale14to15 doubles, hscale16to15 halves).
inline int chromaIdentityS15(const RgbToYuvPlan& p, int raw)
{
    return p.deepSource ? (raw >> 1) : clampInt(2 * raw, -32768, 32767);
}

inline int chromaRange(const RgbToYuvPlan& p, int s)
{
    return p.fullRange ? rangeToJpeg(s, p.chrRangeCoeff, p.chrRangeOffset) : s;
}

// chrToYV12 for one chroma sample on even-width subsampled input (pixel pair
// 2c, 2c+1), then identity hcScale + range.
inline void chromaHalfS15(const RgbToYuvPlan& p, const uint8_t* row, int c, int& u, int& v)
{
    const unsigned r0 = readSample(row, 2 * c, 0, p), r1 = readSample(row, 2 * c + 1, 0, p);
    const unsigned g0 = readSample(row, 2 * c, 1, p), g1 = readSample(row, 2 * c + 1, 1, p);
    const unsigned b0 = readSample(row, 2 * c, 2, p), b1 = readSample(row, 2 * c + 1, 2, p);
    int rawU, rawV;
    if (!p.deepSource)
    {
        const int rs = int(r0 + r1), gs = int(g0 + g1), bs = int(b0 + b1);
        rawU = int16_t((p.ru * rs + p.gu * gs + p.bu * bs + (256 << 15) + (1 << 9)) >> 10);
        rawV = int16_t((p.rv * rs + p.gv * gs + p.bv * bs + (256 << 15) + (1 << 9)) >> 10);
    }
    else
    {
        chromaRaw(p, (r0 + r1 + 1) >> 1, (g0 + g1 + 1) >> 1, (b0 + b1 + 1) >> 1, rawU, rawV);
    }
    u = chromaRange(p, chromaIdentityS15(p, rawU));
    v = chromaRange(p, chromaIdentityS15(p, rawV));
}

// hcScale with the bilinear horizontal chroma filter (odd widths). The deep
// SSE2 hscale16to15 biases samples to signed and adds 0x8000 * 0x4000 back,
// which equals the C hScale16To15 only while the taps sum to exactly 1 << 14.
inline int hScaleS15(const RgbToYuvPlan& p, const int32_t* raw, int c)
{
    const int16_t* f = &p.hFilter[size_t(c) * p.hFilterSize];
    const int32_t* s = raw + p.hFilterPos[c];
    int32_t acc = 0;
    if (!p.deepSource)
    {
        for (int j = 0; j < p.hFilterSize; j++)
            acc += s[j] * f[j];
        return clampInt(acc >> 13, -32768, 32767);
    }
    if (!p.simdHScale16)
    {
        for (int j = 0; j < p.hFilterSize; j++)
            acc += s[j] * f[j];
        return int16_t(std::min(acc >> 15, 32767));
    }
    for (int j = 0; j < p.hFilterSize; j++)
        acc += (s[j] - 32768) * f[j];
    acc = int32_t(uint32_t(acc) + 0x20000000u);
    return clampInt(acc >> 15, -32768, 32767);
}

// yuv2plane1 for one sample (8-bit: dither 64, >> 7; N-bit: round, >> 15-N).
inline int plane1(const RgbToYuvPlan& p, int s)
{
    if (p.outBits == 8)
        return clampInt((s + 64) >> 7, 0, 255);
    const int shift = 15 - p.outBits;
    return clampInt((s + (1 << (shift - 1))) >> shift, 0, (1 << p.outBits) - 1) << p.outShift;
}

// yuv2planeX / yuv2nv12cX / yuv2p01xcX (exact 32-bit) for one sample.
inline int planeXExact(const RgbToYuvPlan& p, const int16_t* const* rows, const int16_t* f,
                       int x)
{
    if (p.outBits == 8)
    {
        uint32_t acc = 64u << 12;
        for (int j = 0; j < p.vFilterSize; j++)
            acc += uint32_t(rows[j][x] * int(f[j]));
        return clampInt(int32_t(acc) >> 19, 0, 255);
    }
    const int shift = 11 + 16 - p.outBits;
    uint32_t acc = 1u << (shift - 1);
    for (int j = 0; j < p.vFilterSize; j++)
        acc += uint32_t(rows[j][x] * int(f[j]));
    return clampInt(int32_t(acc) >> shift, 0, (1 << p.outBits) - 1) << p.outShift;
}

// yuv2yuvX (x86 MMX vfilter): pmulhw per tap in 16-bit, start value
// ((filterSize - 1) * 8 + dither 64) >> 4, psraw 3, packuswb.
inline int planeXApprox(const RgbToYuvPlan& p, const int16_t* const* rows, const int16_t* f,
                        int x)
{
    int16_t acc = int16_t(((p.vFilterSize - 1) * 8 + 64) >> 4);
    for (int j = 0; j < p.vFilterSize; j++)
        acc = int16_t(acc + ((int(f[j]) * rows[j][x]) >> 16));
    return clampInt(acc >> 3, 0, 255);
}

inline void storeSample(const RgbToYuvPlan& p, uint8_t* row, int x, int v)
{
    if (p.outBits == 8)
        row[x] = uint8_t(v);
    else
    {
        row[2 * x] = uint8_t(v);
        row[2 * x + 1] = uint8_t(v >> 8);
    }
}

inline uint8_t* rowPtr(uint8_t* base, int stride, int y) { return base + ptrdiff_t(stride) * y; }

inline bool approxRow(const RgbToYuvPlan& p, int chromaRow)
{
    return p.approxVerticalFilter && (chromaRow << p.chromaShiftY) < p.height - 2;
}

// Writes chroma row @p cy from final 15-bit intermediates (plane1 path).
void writeChromaPlane1Scalar(const RgbToYuvPlan& p, const int16_t* u, const int16_t* v,
                             uint8_t* const dst[4], const int dstStride[4], int cy, int x0)
{
    if (p.semiPlanar)
    {
        uint8_t* row = rowPtr(dst[1], dstStride[1], cy);
        for (int x = x0; x < p.chromaWidth; x++)
        {
            storeSample(p, row, 2 * x, plane1(p, u[x]));
            storeSample(p, row, 2 * x + 1, plane1(p, v[x]));
        }
        return;
    }
    uint8_t* rowU = rowPtr(dst[1], dstStride[1], cy);
    uint8_t* rowV = rowPtr(dst[2], dstStride[2], cy);
    for (int x = x0; x < p.chromaWidth; x++)
    {
        storeSample(p, rowU, x, plane1(p, u[x]));
        storeSample(p, rowV, x, plane1(p, v[x]));
    }
}

// Writes vertically filtered chroma row @p cy from the source-row
// intermediates rowsU/rowsV (vFilterSize each).
void writeChromaVerticalScalar(const RgbToYuvPlan& p, const int16_t* const* rowsU,
                               const int16_t* const* rowsV, uint8_t* const dst[4],
                               const int dstStride[4], int cy, int x0)
{
    if (p.vFilterSize == 1 && !p.semiPlanar)
    {
        writeChromaPlane1Scalar(p, rowsU[0], rowsV[0], dst, dstStride, cy, x0);
        return;
    }
    const int16_t* f = &p.vFilter[size_t(cy) * p.vFilterSize];
    const bool approx = approxRow(p, cy);
    if (p.semiPlanar)
    {
        uint8_t* row = rowPtr(dst[1], dstStride[1], cy);
        for (int x = x0; x < p.chromaWidth; x++)
        {
            storeSample(p, row, 2 * x, planeXExact(p, rowsU, f, x));
            storeSample(p, row, 2 * x + 1, planeXExact(p, rowsV, f, x));
        }
        return;
    }
    uint8_t* rowU = rowPtr(dst[1], dstStride[1], cy);
    uint8_t* rowV = rowPtr(dst[2], dstStride[2], cy);
    for (int x = x0; x < p.chromaWidth; x++)
    {
        storeSample(p, rowU, x, approx ? planeXApprox(p, rowsU, f, x) : planeXExact(p, rowsU, f, x));
        storeSample(p, rowV, x, approx ? planeXApprox(p, rowsV, f, x) : planeXExact(p, rowsV, f, x));
    }
}

void lumaRowScalar(const RgbToYuvPlan& p, const uint8_t* src, uint8_t* dst, int x0)
{
    for (int x = x0; x < p.width; x++)
        storeSample(p, dst, x,
                    plane1(p, lumaS15(p, readSample(src, x, 0, p), readSample(src, x, 1, p),
                                      readSample(src, x, 2, p))));
}

// swscale's x86 luma output for 8- to 10-bit planes stores whole vectors, so
// the luma sample just past an odd row comes out as a copy of the row's last
// one. With horizontally subsampled chroma an encoder reads that sample
// (ljpeg codes whole 2-wide MCUs, yuv4 luma pairs), so it is part of the
// bitstream. swscale switches to its C output (which stops at the width) for
// the last two rows, so those are left alone. Rows [y0, y1), when the stride
// has room for the sample.
void copyLumaEdge(const RgbToYuvPlan& p, uint8_t* dst0, int stride, int y0, int y1)
{
    const int sampleBytes = p.outBits == 8 ? 1 : 2;
    if (!p.lumaEdgeCopy || std::abs(stride) < (p.width + 1) * sampleBytes)
        return;
    for (int y = y0; y < std::min(y1, p.height - 2); y++)
    {
        uint8_t* row = rowPtr(dst0, stride, y);
        std::memcpy(row + size_t(p.width) * sampleBytes,
                    row + size_t(p.width - 1) * sampleBytes, sampleBytes);
    }
}

// Chroma intermediates (after hcScale + range) for one source row, columns
// [c0, chromaWidth). rawScratch holds width + 16 int32 for the odd-width path.
void chromaRowScalar(const RgbToYuvPlan& p, const uint8_t* src, int16_t* u, int16_t* v, int c0,
                     int32_t* rawU, int32_t* rawV)
{
    if (p.halfChroma)
    {
        for (int c = c0; c < p.chromaWidth; c++)
        {
            int cu, cv;
            chromaHalfS15(p, src, c, cu, cv);
            u[c] = int16_t(cu);
            v[c] = int16_t(cv);
        }
        return;
    }
    if (p.chromaShiftX == 0)
    {
        for (int c = c0; c < p.chromaWidth; c++)
        {
            int ru, rv;
            chromaRaw(p, readSample(src, c, 0, p), readSample(src, c, 1, p),
                      readSample(src, c, 2, p), ru, rv);
            u[c] = int16_t(chromaRange(p, chromaIdentityS15(p, ru)));
            v[c] = int16_t(chromaRange(p, chromaIdentityS15(p, rv)));
        }
        return;
    }
    // Odd width into subsampled chroma: full-resolution chroma, then the
    // bilinear hcScale. The filter may read up to hFilterSize - 1 samples past
    // the row with zero weight, so the scratch carries zeroed padding.
    for (int x = 0; x < p.width; x++)
    {
        int ru, rv;
        chromaRaw(p, readSample(src, x, 0, p), readSample(src, x, 1, p), readSample(src, x, 2, p),
                  ru, rv);
        rawU[x] = ru;
        rawV[x] = rv;
    }
    for (int c = c0; c < p.chromaWidth; c++)
    {
        u[c] = int16_t(chromaRange(p, hScaleS15(p, rawU, c)));
        v[c] = int16_t(chromaRange(p, hScaleS15(p, rawV, c)));
    }
}

// Chroma rows needed by luma rows [y0, y1) of a vertically subsampled frame.
void chromaRowRange(const RgbToYuvPlan& p, int y0, int y1, int& k0, int& k1)
{
    k0 = (y0 + (1 << p.chromaShiftY) - 1) >> p.chromaShiftY;
    k1 = std::min((y1 + (1 << p.chromaShiftY) - 1) >> p.chromaShiftY, p.chromaHeight);
}

} // namespace

void sourceRowRange(const RgbToYuvPlan& p, int y0, int y1, int& r0, int& r1)
{
    r0 = y0;
    r1 = y1;
    if (!p.chromaShiftY)
        return;
    int k0, k1;
    chromaRowRange(p, y0, y1, k0, k1);
    if (k0 >= k1)
        return;
    r0 = std::min(r0, p.vFilterPos[k0]);
    r1 = std::max(r1, p.vFilterPos[k1 - 1] + p.vFilterSize);
}

void convertReference(const RgbToYuvPlan& p, const uint8_t* src, ptrdiff_t srcStride,
                      uint8_t* const dst[4], const int dstStride[4], int y0, int y1)
{
    const int cw = p.chromaWidth;
    std::vector<int32_t> rawU(size_t(p.width) + 16, 0), rawV(size_t(p.width) + 16, 0);
    auto srcRow = [&](int y) { return src + srcStride * y; };

    for (int y = y0; y < y1; y++)
        lumaRowScalar(p, srcRow(y), rowPtr(dst[0], dstStride[0], y), 0);
    copyLumaEdge(p, dst[0], dstStride[0], y0, y1);

    if (!p.chromaShiftY)
    {
        std::vector<int16_t> u(cw), v(cw);
        for (int y = y0; y < y1; y++)
        {
            chromaRowScalar(p, srcRow(y), u.data(), v.data(), 0, rawU.data(), rawV.data());
            writeChromaPlane1Scalar(p, u.data(), v.data(), dst, dstStride, y, 0);
        }
        return;
    }

    int k0, k1;
    chromaRowRange(p, y0, y1, k0, k1);
    if (k0 >= k1)
        return;
    const int first = p.vFilterPos[k0];
    const int last = p.vFilterPos[k1 - 1] + p.vFilterSize - 1;
    std::vector<int16_t> rowsU(size_t(last - first + 1) * cw), rowsV(rowsU.size());
    for (int r = first; r <= last; r++)
        chromaRowScalar(p, srcRow(r), &rowsU[size_t(r - first) * cw],
                        &rowsV[size_t(r - first) * cw], 0, rawU.data(), rawV.data());
    std::vector<const int16_t*> tapsU(p.vFilterSize), tapsV(p.vFilterSize);
    for (int k = k0; k < k1; k++)
    {
        for (int j = 0; j < p.vFilterSize; j++)
        {
            tapsU[j] = &rowsU[size_t(p.vFilterPos[k] + j - first) * cw];
            tapsV[j] = &rowsV[size_t(p.vFilterPos[k] + j - first) * cw];
        }
        writeChromaVerticalScalar(p, tapsU.data(), tapsV.data(), dst, dstStride, k, 0);
    }
}

#if NELUX_RGB2YUV_X86
namespace
{

// ---------------------------------------------------------------------------
// AVX2 kernel.
// ---------------------------------------------------------------------------

NELUX_AVX2 inline __m256i loadPair(const uint8_t* lo, const uint8_t* hi)
{
    return _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(lo))),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(hi)), 1);
}

NELUX_AVX2 inline __m256i shuffleMask(int8_t a0, int8_t a1, int8_t a2, int8_t a3, int8_t a4,
                                      int8_t a5, int8_t a6, int8_t a7, int8_t a8, int8_t a9,
                                      int8_t a10, int8_t a11, int8_t a12, int8_t a13,
                                      int8_t a14, int8_t a15)
{
    return _mm256_setr_epi8(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
                            a15, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13,
                            a14, a15);
}

// 16 source pixels as pmaddwd operands: rg = (R, G) word pairs, b = (B, x)
// word pairs (x has weight 0). Group A holds pixels 0-3 | 8-11 (lane 0 |
// lane 1), group B pixels 4-7 | 12-15, which is the order packssdw(A, B)
// turns back into 0..15. Deep sources yield raw unsigned 16-bit samples.
struct Pixels16
{
    __m256i rgA, bA, rgB, bB;
};

struct SrcRgb24
{
    static constexpr int kBytes = 3;
    NELUX_AVX2 static inline Pixels16 load(const uint8_t* s)
    {
        const __m256i a = loadPair(s, s + 24);
        const __m256i b = loadPair(s + 8, s + 32);
        Pixels16 px;
        px.rgA = _mm256_shuffle_epi8(a, shuffleMask(0, -1, 1, -1, 3, -1, 4, -1, 6, -1, 7, -1, 9, -1, 10, -1));
        px.bA = _mm256_shuffle_epi8(a, shuffleMask(2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1));
        px.rgB = _mm256_shuffle_epi8(b, shuffleMask(4, -1, 5, -1, 7, -1, 8, -1, 10, -1, 11, -1, 13, -1, 14, -1));
        px.bB = _mm256_shuffle_epi8(b, shuffleMask(6, -1, -1, -1, 9, -1, -1, -1, 12, -1, -1, -1, 15, -1, -1, -1));
        return px;
    }
};

struct SrcRgba
{
    static constexpr int kBytes = 4;
    NELUX_AVX2 static inline Pixels16 load(const uint8_t* s)
    {
        const __m256i a = loadPair(s, s + 32);
        const __m256i b = loadPair(s + 16, s + 48);
        const __m256i rg = shuffleMask(0, -1, 1, -1, 4, -1, 5, -1, 8, -1, 9, -1, 12, -1, 13, -1);
        const __m256i bl = shuffleMask(2, -1, -1, -1, 6, -1, -1, -1, 10, -1, -1, -1, 14, -1, -1, -1);
        Pixels16 px;
        px.rgA = _mm256_shuffle_epi8(a, rg);
        px.bA = _mm256_shuffle_epi8(a, bl);
        px.rgB = _mm256_shuffle_epi8(b, rg);
        px.bB = _mm256_shuffle_epi8(b, bl);
        return px;
    }
};

// Deep sources: 4 pixels per lane come from two 16-byte loads (first pair
// into the low half of the lane, second pair into the high half).
template <int kPixelBytes>
struct SrcDeep
{
    static constexpr int kBytes = kPixelBytes;
    NELUX_AVX2 static inline void group(const uint8_t* s, const uint8_t* s2, __m256i& rg, __m256i& b)
    {
        // Lane 0: pixels at s; lane 1: pixels at s2 (8 pixels further).
        constexpr int second = kPixelBytes == 6 ? 8 : 16; // second 16-byte load
        constexpr int8_t o2 = kPixelBytes == 6 ? 4 : 0;   // pixel 2 inside it
        const __m256i x0 = loadPair(s, s2);
        const __m256i x1 = loadPair(s + second, s2 + second);
        const int8_t p1 = int8_t(kPixelBytes);
        const __m256i rg0 = shuffleMask(0, 1, 2, 3, p1, p1 + 1, p1 + 2, p1 + 3, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m256i rg1 = shuffleMask(-1, -1, -1, -1, -1, -1, -1, -1, o2, o2 + 1, o2 + 2, o2 + 3,
                                        o2 + p1, o2 + p1 + 1, o2 + p1 + 2, o2 + p1 + 3);
        const __m256i b0 = shuffleMask(4, 5, -1, -1, p1 + 4, p1 + 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m256i b1 = shuffleMask(-1, -1, -1, -1, -1, -1, -1, -1, o2 + 4, o2 + 5, -1, -1,
                                       o2 + p1 + 4, o2 + p1 + 5, -1, -1);
        rg = _mm256_or_si256(_mm256_shuffle_epi8(x0, rg0), _mm256_shuffle_epi8(x1, rg1));
        b = _mm256_or_si256(_mm256_shuffle_epi8(x0, b0), _mm256_shuffle_epi8(x1, b1));
    }
    NELUX_AVX2 static inline Pixels16 load(const uint8_t* s)
    {
        Pixels16 px;
        group(s, s + 8 * kPixelBytes, px.rgA, px.bA);
        group(s + 4 * kPixelBytes, s + 12 * kPixelBytes, px.rgB, px.bB);
        return px;
    }
};

using SrcRgb48 = SrcDeep<6>;
using SrcRgba64 = SrcDeep<8>;

// Per-call constants.
struct Coeffs
{
    __m256i yRG, yB, uRG, uB, vRG, vB;
    __m256i lumRange, lumOffset, chrRange, chrOffset;
};

NELUX_AVX2 inline __m256i wordPair(int32_t lo, int32_t hi)
{
    return _mm256_set1_epi32(int32_t((uint32_t(hi) << 16) | (uint32_t(lo) & 0xFFFF)));
}

NELUX_AVX2 inline Coeffs makeCoeffs(const RgbToYuvPlan& p)
{
    Coeffs c;
    c.yRG = wordPair(p.ry, p.gy);
    c.yB = wordPair(p.by, 0);
    c.uRG = wordPair(p.ru, p.gu);
    c.uB = wordPair(p.bu, 0);
    c.vRG = wordPair(p.rv, p.gv);
    c.vB = wordPair(p.bv, 0);
    c.lumRange = wordPair(p.lumRangeCoeff, 0);
    c.lumOffset = _mm256_set1_epi32(p.lumRangeOffset);
    c.chrRange = wordPair(p.chrRangeCoeff, 0);
    c.chrOffset = _mm256_set1_epi32(p.chrRangeOffset);
    return c;
}

NELUX_AVX2 inline __m256i dot(__m256i rg, __m256i b, __m256i cRG, __m256i cB)
{
    return _mm256_add_epi32(_mm256_madd_epi16(rg, cRG), _mm256_madd_epi16(b, cB));
}

// Deep samples are unsigned; pmaddwd is signed. Flipping the top bit
// subtracts 32768 from every sample, and 32768 * (sum of the three
// coefficients) goes back into the rounding constant.
NELUX_AVX2 inline __m256i bias16(__m256i v)
{
    return _mm256_xor_si256(v, _mm256_set1_epi16(int16_t(0x8000)));
}

NELUX_AVX2 inline int32_t deepBias(int32_t c0, int32_t c1, int32_t c2, uint32_t rnd)
{
    return int32_t(uint32_t(c0 + c1 + c2) * 32768u + rnd);
}

// RangeToJpeg8 on 16 int16 (lane order preserved).
NELUX_AVX2 inline __m256i rangeConvert(__m256i s, __m256i coeff, __m256i offset)
{
    const __m256i zero = _mm256_setzero_si256();
    __m256i lo = _mm256_madd_epi16(_mm256_unpacklo_epi16(s, zero), coeff);
    __m256i hi = _mm256_madd_epi16(_mm256_unpackhi_epi16(s, zero), coeff);
    lo = _mm256_srai_epi32(_mm256_add_epi32(lo, offset), 14);
    hi = _mm256_srai_epi32(_mm256_add_epi32(hi, offset), 14);
    return _mm256_packs_epi32(lo, hi);
}

// One dot product for 16 pixels to a 15-bit intermediate, 8-bit source:
// lumToYV12/chrToYV12 (+rnd, >> 9, int16) followed by the identity hscale's
// doubling (saturating, as packssdw does there).
NELUX_AVX2 inline __m256i s15From8bit(const Pixels16& px, __m256i cRG, __m256i cB, __m256i rnd)
{
    const __m256i a = _mm256_srai_epi32(_mm256_add_epi32(dot(px.rgA, px.bA, cRG, cB), rnd), 9);
    const __m256i b = _mm256_srai_epi32(_mm256_add_epi32(dot(px.rgB, px.bB, cRG, cB), rnd), 9);
    const __m256i s = _mm256_packs_epi32(a, b);
    return _mm256_adds_epi16(s, s);
}

// Same for a deep source (samples already biased): the unsigned 16-bit
// lumToYV12/chrToYV12 result (>> 15) halved by hscale16to15, i.e. >> 16.
NELUX_AVX2 inline __m256i s15From16bit(__m256i rgA, __m256i bA, __m256i rgB, __m256i bB,
                                       __m256i cRG, __m256i cB, __m256i rnd)
{
    const __m256i a = _mm256_srli_epi32(_mm256_add_epi32(dot(rgA, bA, cRG, cB), rnd), 16);
    const __m256i b = _mm256_srli_epi32(_mm256_add_epi32(dot(rgB, bB, cRG, cB), rnd), 16);
    return _mm256_packs_epi32(a, b);
}

// 15-bit luma for 16 pixels, natural order.
template <class Src>
NELUX_AVX2 inline __m256i lumaS15x16(const RgbToYuvPlan& p, const Coeffs& c, const Pixels16& px)
{
    __m256i y;
    if constexpr (Src::kBytes <= 4)
        y = s15From8bit(px, c.yRG, c.yB, _mm256_set1_epi32(0x80100));
    else
        y = s15From16bit(bias16(px.rgA), bias16(px.bA), bias16(px.rgB), bias16(px.bB), c.yRG,
                         c.yB, _mm256_set1_epi32(deepBias(p.ry, p.gy, p.by, 0x8004000u)));
    if (p.fullRange)
        y = rangeConvert(y, c.lumRange, c.lumOffset);
    return y;
}

// Full-resolution 15-bit chroma (4:4:4) for 16 pixels.
template <class Src>
NELUX_AVX2 inline void chromaS15x16(const RgbToYuvPlan& p, const Coeffs& c, const Pixels16& px,
                                    __m256i& u, __m256i& v)
{
    if constexpr (Src::kBytes <= 4)
    {
        const __m256i rnd = _mm256_set1_epi32(0x400100);
        u = s15From8bit(px, c.uRG, c.uB, rnd);
        v = s15From8bit(px, c.vRG, c.vB, rnd);
    }
    else
    {
        const __m256i rgA = bias16(px.rgA), bA = bias16(px.bA);
        const __m256i rgB = bias16(px.rgB), bB = bias16(px.bB);
        u = s15From16bit(rgA, bA, rgB, bB, c.uRG, c.uB,
                         _mm256_set1_epi32(deepBias(p.ru, p.gu, p.bu, 0x40004000u)));
        v = s15From16bit(rgA, bA, rgB, bB, c.vRG, c.vB,
                         _mm256_set1_epi32(deepBias(p.rv, p.gv, p.bv, 0x40004000u)));
    }
    if (p.fullRange)
    {
        u = rangeConvert(u, c.chrRange, c.chrOffset);
        v = rangeConvert(v, c.chrRange, c.chrOffset);
    }
}

// rgb24ToUV_half sums the pixel pair before the dot product; the dot is
// linear, so adding the two per-pixel dots gives the same integer.
NELUX_AVX2 inline __m256i halfDot8bit(const Pixels16& px, __m256i cRG, __m256i cB)
{
    const __m256i pairs =
        _mm256_hadd_epi32(dot(px.rgA, px.bA, cRG, cB), dot(px.rgB, px.bB, cRG, cB));
    return _mm256_srai_epi32(_mm256_add_epi32(pairs, _mm256_set1_epi32((256 << 15) + (1 << 9))),
                             10);
}

NELUX_AVX2 inline __m256i pickPairs(__m256i a, __m256i b, bool odd)
{
    const __m256 fa = _mm256_castsi256_ps(a), fb = _mm256_castsi256_ps(b);
    return _mm256_castps_si256(odd ? _mm256_shuffle_ps(fa, fb, _MM_SHUFFLE(3, 1, 3, 1))
                                   : _mm256_shuffle_ps(fa, fb, _MM_SHUFFLE(2, 0, 2, 0)));
}

// Horizontally halved 15-bit chroma (even widths) for 16 pixels -> 8
// chroma samples: low 128 bits U, high 128 bits V.
template <class Src>
NELUX_AVX2 inline __m256i chromaHalfS15x16(const RgbToYuvPlan& p, const Coeffs& c,
                                           const Pixels16& px)
{
    __m256i uv;
    if constexpr (Src::kBytes <= 4)
    {
        uv = _mm256_packs_epi32(halfDot8bit(px, c.uRG, c.uB), halfDot8bit(px, c.vRG, c.vB));
        // lane 0: u0-3 v0-3, lane 1: u4-7 v4-7 -> u0-7 | v0-7
        uv = _mm256_permute4x64_epi64(uv, _MM_SHUFFLE(3, 1, 2, 0));
        uv = _mm256_adds_epi16(uv, uv);
    }
    else
    {
        // rgb48ToUV_half averages each channel ((a + b + 1) >> 1 == pavgw)
        // before the dot product. Even/odd pixels of groups A and B come out
        // as pixels 0,2,4,6 | 8,10,12,14 and 1,3,5,7 | 9,11,13,15.
        const __m256i rg = bias16(_mm256_avg_epu16(pickPairs(px.rgA, px.rgB, false),
                                                   pickPairs(px.rgA, px.rgB, true)));
        const __m256i b = bias16(_mm256_avg_epu16(pickPairs(px.bA, px.bB, false),
                                                  pickPairs(px.bA, px.bB, true)));
        const __m256i u = _mm256_srli_epi32(
            _mm256_add_epi32(dot(rg, b, c.uRG, c.uB),
                             _mm256_set1_epi32(deepBias(p.ru, p.gu, p.bu, 0x40004000u))),
            16);
        const __m256i v = _mm256_srli_epi32(
            _mm256_add_epi32(dot(rg, b, c.vRG, c.vB),
                             _mm256_set1_epi32(deepBias(p.rv, p.gv, p.bv, 0x40004000u))),
            16);
        uv = _mm256_permute4x64_epi64(_mm256_packs_epi32(u, v), _MM_SHUFFLE(3, 1, 2, 0));
    }
    if (p.fullRange)
        uv = rangeConvert(uv, c.chrRange, c.chrOffset);
    return uv;
}

// yuv2plane1 for 16 samples, stored at dst (8-bit: 16 bytes, N-bit: 32).
NELUX_AVX2 inline void storePlane1x16(const RgbToYuvPlan& p, __m256i s, uint8_t* dst)
{
    if (p.outBits == 8)
    {
        const __m256i o = _mm256_srai_epi16(_mm256_adds_epi16(s, _mm256_set1_epi16(64)), 7);
        const __m256i b = _mm256_permute4x64_epi64(_mm256_packus_epi16(o, o), _MM_SHUFFLE(0, 0, 2, 0));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_castsi256_si128(b));
        return;
    }
    const int shift = 15 - p.outBits;
    __m256i o = _mm256_sra_epi16(_mm256_adds_epi16(s, _mm256_set1_epi16(int16_t(1 << (shift - 1)))),
                                 _mm_cvtsi32_si128(shift));
    o = _mm256_max_epi16(o, _mm256_setzero_si256());
    o = _mm256_min_epi16(o, _mm256_set1_epi16(int16_t((1 << p.outBits) - 1)));
    o = _mm256_sll_epi16(o, _mm_cvtsi32_si128(p.outShift));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), o);
}

// One source row: luma written to yDst (when non-null), chroma intermediates
// to u/v (when non-null). rawU/rawV: odd-width scratch (width + 16 int32).
template <class Src>
NELUX_AVX2 void rowPass(const RgbToYuvPlan& p, const Coeffs& c, const uint8_t* src,
                        uint8_t* yDst, int16_t* u, int16_t* v, int32_t* rawU, int32_t* rawV)
{
    const int w16 = p.width & ~15;
    const int sampleBytes = p.outBits == 8 ? 1 : 2;
    const bool wantChroma = u != nullptr;
    const bool fullChroma = wantChroma && !p.halfChroma && p.chromaShiftX == 0;
    const bool halfChroma = wantChroma && p.halfChroma;

    for (int x = 0; x < w16; x += 16)
    {
        const Pixels16 px = Src::load(src + size_t(x) * Src::kBytes);
        if (yDst)
            storePlane1x16(p, lumaS15x16<Src>(p, c, px), yDst + size_t(x) * sampleBytes);
        if (fullChroma)
        {
            __m256i cu, cv;
            chromaS15x16<Src>(p, c, px, cu, cv);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(u + x), cu);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(v + x), cv);
        }
        else if (halfChroma)
        {
            const __m256i uv = chromaHalfS15x16<Src>(p, c, px);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(u + x / 2), _mm256_castsi256_si128(uv));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(v + x / 2), _mm256_extracti128_si256(uv, 1));
        }
    }
    if (yDst)
        lumaRowScalar(p, src, yDst, w16);
    if (!wantChroma)
        return;
    // Vector part done above; the rest (and all of an odd-width subsampled
    // row, whose bilinear hcScale stays scalar: encoders mostly reject odd
    // sizes for subsampled formats) goes through the model.
    chromaRowScalar(p, src, u, v, fullChroma ? w16 : (halfChroma ? w16 / 2 : 0), rawU, rawV);
}

// Chroma plane1 output (no vertical filter) for one chroma row.
NELUX_AVX2 void writeChromaPlane1(const RgbToYuvPlan& p, const int16_t* u, const int16_t* v,
                                  uint8_t* const dst[4], const int dstStride[4], int cy)
{
    const int cw = p.chromaWidth;
    const int w16 = cw & ~15;
    const int sampleBytes = p.outBits == 8 ? 1 : 2;
    if (p.semiPlanar)
    {
        // Semi-planar output always goes through the vertical filter (nv12cX
        // runs even at filter size 1); nothing in the table reaches here.
        writeChromaPlane1Scalar(p, u, v, dst, dstStride, cy, 0);
        return;
    }
    uint8_t* rowU = rowPtr(dst[1], dstStride[1], cy);
    uint8_t* rowV = rowPtr(dst[2], dstStride[2], cy);
    for (int x = 0; x < w16; x += 16)
    {
        storePlane1x16(p, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + x)),
                       rowU + size_t(x) * sampleBytes);
        storePlane1x16(p, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v + x)),
                       rowV + size_t(x) * sampleBytes);
    }
    writeChromaPlane1Scalar(p, u, v, dst, dstStride, cy, w16);
}

// Exact 32-bit vertical filter for 16 samples: round + sum(c_j * s_j) >>
// shift, as two dword halves (pixels 0-3|8-11 and 4-7|12-15).
NELUX_AVX2 inline void verticalExact16(const RgbToYuvPlan& p, const int16_t* const* rows,
                                       const int16_t* f, int x, int shift, __m256i& lo, __m256i& hi)
{
    lo = hi = _mm256_set1_epi32(1 << (shift - 1));
    for (int j = 0; j < p.vFilterSize; j += 2)
    {
        const bool pair = j + 1 < p.vFilterSize;
        const __m256i s0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[j] + x));
        const __m256i s1 = pair ? _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[j + 1] + x))
                                : _mm256_setzero_si256();
        const __m256i cf = wordPair(f[j], pair ? f[j + 1] : 0);
        lo = _mm256_add_epi32(lo, _mm256_madd_epi16(_mm256_unpacklo_epi16(s0, s1), cf));
        hi = _mm256_add_epi32(hi, _mm256_madd_epi16(_mm256_unpackhi_epi16(s0, s1), cf));
    }
    const __m128i sh = _mm_cvtsi32_si128(shift);
    lo = _mm256_sra_epi32(lo, sh);
    hi = _mm256_sra_epi32(hi, sh);
}

// 16 vertically filtered samples as 8-bit (low 128 bits) or N-bit words.
NELUX_AVX2 inline __m256i verticalSamples16(const RgbToYuvPlan& p, const int16_t* const* rows,
                                            const int16_t* f, int x, bool approx)
{
    if (approx)
    {
        __m256i acc = _mm256_set1_epi16(int16_t(((p.vFilterSize - 1) * 8 + 64) >> 4));
        for (int j = 0; j < p.vFilterSize; j++)
            acc = _mm256_add_epi16(
                acc, _mm256_mulhi_epi16(_mm256_set1_epi16(f[j]),
                                        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rows[j] + x))));
        acc = _mm256_srai_epi16(acc, 3);
        return _mm256_permute4x64_epi64(_mm256_packus_epi16(acc, acc), _MM_SHUFFLE(0, 0, 2, 0));
    }
    __m256i lo, hi;
    if (p.outBits == 8)
    {
        verticalExact16(p, rows, f, x, 19, lo, hi);
        const __m256i w = _mm256_packs_epi32(lo, hi);
        return _mm256_permute4x64_epi64(_mm256_packus_epi16(w, w), _MM_SHUFFLE(0, 0, 2, 0));
    }
    verticalExact16(p, rows, f, x, 11 + 16 - p.outBits, lo, hi);
    __m256i w = _mm256_packus_epi32(lo, hi);
    w = _mm256_min_epi16(w, _mm256_set1_epi16(int16_t((1 << p.outBits) - 1)));
    return _mm256_sll_epi16(w, _mm_cvtsi32_si128(p.outShift));
}

NELUX_AVX2 void writeChromaVertical(const RgbToYuvPlan& p, const int16_t* const* rowsU,
                                    const int16_t* const* rowsV, uint8_t* const dst[4],
                                    const int dstStride[4], int cy)
{
    if (p.vFilterSize == 1 && !p.semiPlanar)
    {
        writeChromaPlane1(p, rowsU[0], rowsV[0], dst, dstStride, cy);
        return;
    }
    const int cw = p.chromaWidth;
    const int w16 = cw & ~15;
    const int16_t* f = &p.vFilter[size_t(cy) * p.vFilterSize];
    const bool approx = approxRow(p, cy);
    if (p.semiPlanar)
    {
        uint8_t* row = rowPtr(dst[1], dstStride[1], cy);
        for (int x = 0; x < w16; x += 16)
        {
            const __m256i su = verticalSamples16(p, rowsU, f, x, false);
            const __m256i sv = verticalSamples16(p, rowsV, f, x, false);
            if (p.outBits == 8)
            {
                const __m128i a = _mm256_castsi256_si128(su), b = _mm256_castsi256_si128(sv);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(row + 2 * x), _mm_unpacklo_epi8(a, b));
                _mm_storeu_si128(reinterpret_cast<__m128i*>(row + 2 * x + 16), _mm_unpackhi_epi8(a, b));
            }
            else
            {
                const __m256i lo = _mm256_unpacklo_epi16(su, sv);
                const __m256i hi = _mm256_unpackhi_epi16(su, sv);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(row + 4 * x),
                                    _mm256_permute2x128_si256(lo, hi, 0x20));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(row + 4 * x + 32),
                                    _mm256_permute2x128_si256(lo, hi, 0x31));
            }
        }
        writeChromaVerticalScalar(p, rowsU, rowsV, dst, dstStride, cy, w16);
        return;
    }
    uint8_t* rowU = rowPtr(dst[1], dstStride[1], cy);
    uint8_t* rowV = rowPtr(dst[2], dstStride[2], cy);
    for (int x = 0; x < w16; x += 16)
    {
        const __m256i su = verticalSamples16(p, rowsU, f, x, approx);
        const __m256i sv = verticalSamples16(p, rowsV, f, x, approx);
        if (p.outBits == 8)
        {
            _mm_storeu_si128(reinterpret_cast<__m128i*>(rowU + x), _mm256_castsi256_si128(su));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(rowV + x), _mm256_castsi256_si128(sv));
        }
        else
        {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(rowU + 2 * x), su);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(rowV + 2 * x), sv);
        }
    }
    writeChromaVerticalScalar(p, rowsU, rowsV, dst, dstStride, cy, w16);
}

// State of one convertAvx2 call. Kept as a struct with AVX2-attributed
// members rather than lambdas: GCC and Clang do not carry a function's
// target("avx2") into the lambdas it defines.
template <class Src>
struct FrameJob
{
    const RgbToYuvPlan& p;
    Coeffs c;
    const uint8_t* src;
    ptrdiff_t srcStride;
    uint8_t* const* dst;
    const int* dstStride;
    int y0, y1;
    int cwPad;
    int32_t* rawU = nullptr;
    int32_t* rawV = nullptr;

    // Vertically subsampled frames: a ring of chroma intermediates for the
    // source rows the vertical filter taps, each row computed once.
    int ringRows = 0;
    int16_t* ring = nullptr;
    int* ringTag = nullptr;
    int nextLuma = 0; // luma rows are written in order as rows go through

    const uint8_t* srcRow(int y) const { return src + srcStride * y; }
    uint8_t* lumaRow(int y) const { return rowPtr(dst[0], dstStride[0], y); }
    int16_t* ringU(int slot) const { return ring + size_t(slot) * 2 * cwPad; }

    NELUX_AVX2 void lumaOnly(int y)
    {
        rowPass<Src>(p, c, srcRow(y), lumaRow(y), nullptr, nullptr, rawU, rawV);
    }

    NELUX_AVX2 void ensureRow(int r)
    {
        const int slot = r % ringRows;
        if (ringTag[slot] == r)
            return;
        for (; nextLuma < r && nextLuma < y1; nextLuma++)
            lumaOnly(nextLuma);
        uint8_t* yDst = nullptr;
        if (r == nextLuma && r < y1)
        {
            yDst = lumaRow(r);
            nextLuma++;
        }
        rowPass<Src>(p, c, srcRow(r), yDst, ringU(slot), ringU(slot) + cwPad, rawU, rawV);
        ringTag[slot] = r;
    }
};

template <class Src>
NELUX_AVX2 void convertAvx2(const RgbToYuvPlan& p, const uint8_t* src, ptrdiff_t srcStride,
                            uint8_t* const dst[4], const int dstStride[4], int y0, int y1)
{
    FrameJob<Src> job{p, makeCoeffs(p), src, srcStride, dst, dstStride, y0, y1,
                      ((p.chromaWidth + 15) & ~15) + 16};
    const bool oddWidth = !p.halfChroma && p.chromaShiftX != 0;
    std::vector<int32_t> raw(oddWidth ? 2 * (size_t(p.width) + 16) : 0, 0);
    if (oddWidth)
    {
        job.rawU = raw.data();
        job.rawV = raw.data() + p.width + 16;
    }

    if (!p.chromaShiftY)
    {
        std::vector<int16_t> uv(2 * size_t(job.cwPad));
        for (int y = y0; y < y1; y++)
        {
            rowPass<Src>(p, job.c, job.srcRow(y), job.lumaRow(y), uv.data(),
                         uv.data() + job.cwPad, job.rawU, job.rawV);
            writeChromaPlane1(p, uv.data(), uv.data() + job.cwPad, dst, dstStride, y);
        }
        _mm256_zeroupper();
        copyLumaEdge(p, dst[0], dstStride[0], y0, y1);
        return;
    }

    job.ringRows = p.vFilterSize + 2;
    std::vector<int16_t> ring(2 * size_t(job.ringRows) * job.cwPad);
    std::vector<int> ringTag(job.ringRows, -1);
    job.ring = ring.data();
    job.ringTag = ringTag.data();
    job.nextLuma = y0;
    std::vector<const int16_t*> tapsU(p.vFilterSize), tapsV(p.vFilterSize);

    int k0, k1;
    chromaRowRange(p, y0, y1, k0, k1);
    for (int k = k0; k < k1; k++)
    {
        for (int j = 0; j < p.vFilterSize; j++)
            job.ensureRow(p.vFilterPos[k] + j);
        // The ring holds more rows than the filter has taps, so computing the
        // later taps never evicted an earlier one.
        for (int j = 0; j < p.vFilterSize; j++)
        {
            tapsU[j] = job.ringU((p.vFilterPos[k] + j) % job.ringRows);
            tapsV[j] = tapsU[j] + job.cwPad;
        }
        writeChromaVertical(p, tapsU.data(), tapsV.data(), dst, dstStride, k);
    }
    for (; job.nextLuma < y1; job.nextLuma++)
        job.lumaOnly(job.nextLuma);
    _mm256_zeroupper();
    copyLumaEdge(p, dst[0], dstStride[0], y0, y1);
}

RgbToYuvFn kernelFor(AVPixelFormat srcFmt)
{
    switch (srcFmt)
    {
    case AV_PIX_FMT_RGB24:    return convertAvx2<SrcRgb24>;
    case AV_PIX_FMT_RGBA:     return convertAvx2<SrcRgba>;
    case AV_PIX_FMT_RGB48LE:  return convertAvx2<SrcRgb48>;
    case AV_PIX_FMT_RGBA64LE: return convertAvx2<SrcRgba64>;
    default:                  return nullptr;
    }
}

} // namespace

namespace
{

struct DstInfo
{
    int bits, shiftX, shiftY;
    bool semiPlanar;
    int outShift;
};

bool dstInfo(AVPixelFormat fmt, DstInfo& d)
{
    switch (fmt)
    {
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:    d = {8, 1, 1, false, 0}; return true;
    case AV_PIX_FMT_NV12:        d = {8, 1, 1, true, 0}; return true;
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:    d = {8, 1, 0, false, 0}; return true;
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:    d = {8, 0, 0, false, 0}; return true;
    case AV_PIX_FMT_YUV420P10LE: d = {10, 1, 1, false, 0}; return true;
    case AV_PIX_FMT_YUV422P10LE: d = {10, 1, 0, false, 0}; return true;
    case AV_PIX_FMT_YUV444P10LE: d = {10, 0, 0, false, 0}; return true;
    case AV_PIX_FMT_YUV420P12LE: d = {12, 1, 1, false, 0}; return true;
    case AV_PIX_FMT_YUV422P12LE: d = {12, 1, 0, false, 0}; return true;
    case AV_PIX_FMT_YUV444P12LE: d = {12, 0, 0, false, 0}; return true;
    case AV_PIX_FMT_P010LE:      d = {10, 1, 1, true, 6}; return true;
    default:                     return false;
    }
}

} // namespace
#endif // NELUX_RGB2YUV_X86

std::unique_ptr<RgbToYuvPlan> makePlan(AVPixelFormat srcFmt, AVPixelFormat dstFmt,
                                       int swsColorspace, int dstRange, int width, int height,
                                       int swsFlags)
{
    // Any other flag set (ACCURATE_RND, BITEXACT, FULL_CHR_H_INP, another
    // scaler...) changes swscale's path; leave those to swscale.
#if !NELUX_RGB2YUV_X86
    (void)srcFmt, (void)dstFmt, (void)swsColorspace, (void)dstRange, (void)width, (void)height,
        (void)swsFlags;
    return nullptr;
#else
    if (swsFlags != SWS_BILINEAR || width <= 0 || height <= 0 || (dstRange != 0 && dstRange != 1))
        return nullptr;

    auto p = std::make_unique<RgbToYuvPlan>();
    p->srcFmt = srcFmt;
    p->dstFmt = dstFmt;
    p->width = width;
    p->height = height;
    switch (srcFmt)
    {
    case AV_PIX_FMT_RGB24:    p->srcPixelBytes = 3; p->deepSource = false; break;
    case AV_PIX_FMT_RGBA:     p->srcPixelBytes = 4; p->deepSource = false; break;
    case AV_PIX_FMT_RGB48LE:  p->srcPixelBytes = 6; p->deepSource = true; break;
    case AV_PIX_FMT_RGBA64LE: p->srcPixelBytes = 8; p->deepSource = true; break;
    default:                  return nullptr;
    }

    DstInfo d;
    if (!dstInfo(dstFmt, d))
        return nullptr;
    // A 16-bit source into an 8-bit destination is ordered-dithered by
    // swscale (should_dither); not modelled.
    if (p->deepSource && d.bits == 8)
        return nullptr;
    p->outBits = d.bits;
    p->outShift = d.outShift;
    p->semiPlanar = d.semiPlanar;
    p->chromaShiftX = d.shiftX;
    p->chromaShiftY = d.shiftY;
    p->chromaWidth = (width + (1 << d.shiftX) - 1) >> d.shiftX;
    p->chromaHeight = (height + (1 << d.shiftY) - 1) >> d.shiftY;
    p->halfChroma = d.shiftX == 1 && (width & 1) == 0;
    // 12-bit planes and p010 go through swscale's C output, which stops at
    // the width.
    p->lumaEdgeCopy = d.shiftX != 0 && (width & 1) != 0 && d.bits <= 10 && d.outShift == 0;

    fillRgb2YuvTable(sws_getCoefficients(swsColorspace), *p);

    if (dstRange == 1)
    {
        uint32_t coeff;
        int64_t offset;
        p->fullRange = true;
        solveRangeConvert(d.bits, uint16_t(235u << (d.bits - 8)), coeff, offset);
        // The SIMD converter multiplies by the coefficient's low word as a
        // signed value; past 32767 it and the C path part ways.
        if (coeff > 32767)
            return nullptr;
        p->lumRangeCoeff = int32_t(coeff);
        p->lumRangeOffset = int32_t(uint32_t(uint64_t(offset)));
        solveRangeConvert(d.bits, uint16_t(240u << (d.bits - 8)), coeff, offset);
        if (coeff > 32767)
            return nullptr;
        p->chrRangeCoeff = int32_t(coeff);
        p->chrRangeOffset = int32_t(uint32_t(uint64_t(offset)));
    }

    // swscale picks filter alignment and its SIMD output paths from FFmpeg's
    // own CPU flags at init; read them the same way so the plan follows
    // swscale even if those flags were forced down (av_force_cpu_flags).
    const int cpuFlags = av_get_cpu_flags();
    const bool cpuMmx = (cpuFlags & AV_CPU_FLAG_MMX) != 0;
    p->simdHScale16 = (cpuFlags & AV_CPU_FLAG_SSE2) != 0;

    // get_local_pos(): default chroma siting gives 128 on both sides.
    const int chromaPos = 128;
    if (d.shiftX && !p->halfChroma)
    {
        const int xInc = int(((int64_t(width) << 16) + (p->chromaWidth >> 1)) / p->chromaWidth);
        if (!initFilterBilinear(xInc, width, p->chromaWidth, cpuMmx ? 4 : 1, 1 << 14, chromaPos,
                                chromaPos, cpuMmx, p->hFilter, p->hFilterPos, p->hFilterSize))
            return nullptr;
    }
    if (d.shiftY)
    {
        const int yInc = int(((int64_t(height) << 16) + (p->chromaHeight >> 1)) / p->chromaHeight);
        if (!initFilterBilinear(yInc, height, p->chromaHeight, cpuMmx ? 2 : 1, 1 << 12, chromaPos,
                                chromaPos, cpuMmx, p->vFilter, p->vFilterPos, p->vFilterSize))
            return nullptr;
        // ff_sws_init_swscale_x86: use_mmx_vfilter (yuv2yuvX) needs MMXEXT +
        // SSE2 and an 8-bit, non-semi-planar destination.
        p->approxVerticalFilter = d.bits == 8 && !d.semiPlanar && p->vFilterSize > 1 &&
                                  (cpuFlags & AV_CPU_FLAG_MMXEXT) && (cpuFlags & AV_CPU_FLAG_SSE2);
    }

    p->fn = cpuHasAvx2() ? kernelFor(srcFmt) : convertReference;
    return p;
#endif
}

RgbToYuvFn canHandle(AVPixelFormat srcFmt, AVPixelFormat dstFmt, int swsColorspace,
                     int dstRange, int width, int height, int swsFlags)
{
    if (!cpuHasAvx2())
        return nullptr;
    const auto plan = makePlan(srcFmt, dstFmt, swsColorspace, dstRange, width, height, swsFlags);
    return plan ? plan->fn : nullptr;
}

} // namespace simd
} // namespace cpu
} // namespace conversion
} // namespace nelux
