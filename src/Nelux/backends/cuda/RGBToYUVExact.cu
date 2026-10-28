/*
 * RGBToYUVExact.cu - CUDA RGB -> YUV kernels, byte-exact with nelux's CPU
 * swscale path (RGBToAutoConverter: legacy context, SWS_BILINEAR, same
 * size, libswscale 10.1.102 on x86-64). See RGBToYUVExact.cuh.
 *
 * What swscale runs for these pairs, stage by stage, and what each kernel
 * reproduces (input.c / swscale.c / output.c / x86 references are 9.0.2):
 *
 *  luma   8-bit src: Y14 = (ry*R + gy*G + by*B + (32<<14) + (1<<8)) >> 9
 *                    (rgb24ToY, the AVX2 reader is memcmp-exact with C)
 *         16-bit src: Y16 = (ry*R + gy*G + by*B + (0x2001<<14)) >> 15, in
 *                    unsigned arithmetic (rgb48ToY_c, no SIMD reader)
 *         identity hscale: Y15 = 2*Y14 or Y16 >> 1 (hscale14to15/16to15)
 *  chroma even-width 4:2:x reads pixel PAIRS (chrSrcHSubSample = 1):
 *                    8-bit: U14 = (ru*(R0+R1) + ... + (256<<15) + (1<<9)) >> 10
 *                    16-bit: averages each channel ((a+b+1)>>1) first
 *         otherwise full resolution: U14 = (ru*R + ... + (256<<14) + (1<<8)) >> 9
 *         identity hscale doubles it; an odd-width 4:2:x destination runs a
 *         real bilinear hscale (its taps come from the host plan)
 *  range  full-range output adds lumRangeToJpeg / chrRangeToJpeg on the
 *         15-bit values: min((v*coeff + offset) >> 14, 32767)
 *  vertical
 *         4:4:4 / 4:2:2 and all luma: yuv2plane1, (v + 64) >> 7 at 8 bits
 *         (dither is the flat 64 for an 8-bit source), (v + 16) >> 5 at 10
 *         4:2:0 chroma: the 2:1 bilinear filter from the plan, evaluated as
 *           - yuv420p/yuvj420p: yuv2yuvX_avx2 (use_mmx_vfilter), a pmulhw
 *             approximation: ((((taps-1)*8 + 64) >> 4) + sum((c*v) >> 16)) >> 3,
 *             EXCEPT the last chroma row, which swscale emits through the C
 *             yuv2planeX_8_c because dstY >= dstH - 2 there. This is the only
 *             stage where swscale's C and SIMD output differ (by 1 LSB);
 *           - nv12: yuv2nv12cX (avx2 == C): ((64<<12) + sum(c*v)) >> 19;
 *           - 10-bit planar / p010: ((1<<16) + sum(c*v)) >> 17.
 *
 * Verified against the real swscale-10.dll by a standalone parity suite
 * (all 2^24 colours, all 511^3 pixel-pair sums, every W,H in 1..34, 360
 * random odd sizes up to 4097x2161, real frames, all matrices x ranges, a
 * context reused across frames, and swscale's SSE2 / SSSE3 / SSE4.2 / AVX
 * selections as well as AVX2) and against nelux's own CPU encode path
 * (rawvideo packets). A plain-C or non-x86 libswscale differs in the 4:2:0
 * yuv420p chroma rows; RgbToYuvExactConverter::selfTest catches that.
 * Kernels never read global state: all constants arrive by value.
 *
 * A float32 / float16 / bfloat16 frame is read in place and each sample
 * narrowed in the load (floatSample) to the uint8 / uint16 that encode_frame's
 * torch normalisation writes; the swscale math after it is the integer one.
 */

#ifdef NELUX_ENABLE_CUDA

#include <Nelux/backends/cuda/RGBToYUVExact.cuh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace nelux::backends::cuda
{
// Named (not anonymous) so nvcc does not warn about the traits a given
// target leaves unused.
namespace exact_kernel_detail
{

//------------------------------------------------------------------------------
// Destination traits
//------------------------------------------------------------------------------
template <ExactYuvTarget T> struct TargetTraits
{
    static constexpr int kBits =
        (T == ExactYuvTarget::Yuv420p10 || T == ExactYuvTarget::Yuv422p10 ||
         T == ExactYuvTarget::Yuv444p10 || T == ExactYuvTarget::P010) ? 10 : 8;
    static constexpr bool kSemiPlanar = T == ExactYuvTarget::Nv12 || T == ExactYuvTarget::P010;
    static constexpr int kShiftX = (T == ExactYuvTarget::Yuv444p || T == ExactYuvTarget::Yuv444p10) ? 0 : 1;
    static constexpr int kShiftY = (T == ExactYuvTarget::Yuv420p || T == ExactYuvTarget::Nv12 ||
                                    T == ExactYuvTarget::Yuv420p10 || T == ExactYuvTarget::P010) ? 1 : 0;
    // p010 keeps its 10 bits in the high end of each 16-bit word.
    static constexpr int kMsbShift = T == ExactYuvTarget::P010 ? 6 : 0;
    // 8-bit planar output goes through yuv2yuvX (use_mmx_vfilter = 1);
    // semi-planar and >8-bit destinations do not.
    static constexpr bool kMmxVFilter = kBits == 8 && !kSemiPlanar;
};

//------------------------------------------------------------------------------
// Source element traits. An integer source (uint8_t, uint16_t) is read as
// the samples swscale's math takes. A float frame is FloatSamples<S>: its
// elements (float32, float16 or bfloat16, the source's ExactElement, told
// apart at run time) are narrowed to the sample type S the encoder would
// have staged. One kernel per target serves all three float types and any
// layout, read through the strides: a vectorised load per type and layout
// would multiply the module's device code several times over.
//------------------------------------------------------------------------------
template <typename S> struct FloatSamples
{
};

template <typename E> struct ElementTraits
{
    using Sample = E;
    static constexpr bool kFloat = false;
};

template <typename S> struct ElementTraits<FloatSamples<S>>
{
    using Sample = S;
    static constexpr bool kFloat = true;
};

} // namespace exact_kernel_detail

namespace
{
using exact_kernel_detail::ElementTraits;
using exact_kernel_detail::FloatSamples;
using exact_kernel_detail::TargetTraits;

constexpr int kPx = 8; // horizontal pixels per thread

// 16-bit samples (Rgb48 / Rgba64): swscale's 16-bit input readers.
template <typename E>
constexpr bool kDeepSample = sizeof(typename ElementTraits<E>::Sample) == 2;

__device__ __forceinline__ int clampInt(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

//------------------------------------------------------------------------------
// Float narrowing: what encode_frame's torch ops give, one sample at a time
//------------------------------------------------------------------------------

// 8 bits: (x * 255.0f).clamp(0, 255).to(uint8); 16 bits:
// (x * 65535.0f).round().clamp(0, 65535).to(uint16). One correctly rounded
// multiply (__fmul_rn: never contracted into an FMA), round-half-even for
// round(), truncation for the cast. torch's clamp keeps a NaN and its cast
// makes it 0; fmaxf returns the non-NaN operand, so NaN is 0 here too.
template <bool Deep>
__device__ __forceinline__ int narrowFloat(float x)
{
    if constexpr (Deep)
        return int(fminf(fmaxf(rintf(__fmul_rn(x, 65535.0f)), 0.0f), 65535.0f));
    else
        return int(fminf(fmaxf(__fmul_rn(x, 255.0f), 0.0f), 255.0f));
}

// Float element @p i of @p data, narrowed to the sample the YUV math reads.
// float16 and bfloat16 widen to float32 exactly, as x.to(float32) does.
template <typename E>
__device__ __forceinline__ int floatSample(const void* data, int64_t i, ExactElement element)
{
    float x;
    if (element == ExactElement::Float32)
        x = __ldg(static_cast<const float*>(data) + i);
    else if (element == ExactElement::Float16)
        x = __half2float(__ushort_as_half(__ldg(static_cast<const unsigned short*>(data) + i)));
    else // bfloat16 is float32's high half
        x = __uint_as_float(unsigned(__ldg(static_cast<const unsigned short*>(data) + i)) << 16);
    return narrowFloat<kDeepSample<E>>(x);
}

//------------------------------------------------------------------------------
// Per-sample swscale math (15-bit intermediates)
//------------------------------------------------------------------------------

// Luma after the identity hscale and the optional range pass.
template <typename E>
__device__ __forceinline__ int lumaY15(const RgbToYuvExactParams& p, int r, int g, int b)
{
    int y15;
    if constexpr (!kDeepSample<E>)
    {
        const int y14 = (p.ry * r + p.gy * g + p.by * b + (32 << 14) + (1 << 8)) >> 9;
        y15 = min(2 * y14, 32767);
    }
    else
    {
        const unsigned y16 = (unsigned(p.ry) * unsigned(r) + unsigned(p.gy) * unsigned(g) +
                              unsigned(p.by) * unsigned(b) + (0x2001u << 14)) >> 15;
        y15 = int(y16 & 0xFFFFu) >> 1;
    }
    if (p.rangeConvert)
        y15 = min((y15 * p.lumRangeCoeff + p.lumRangeOffset) >> 14, 32767);
    return y15;
}

// Chroma reader output (before hscale): the int16 buffer value swscale's
// chrToYV12 writes. 8-bit sources give U14 (pair or single), 16-bit U16.
template <typename E>
__device__ __forceinline__ void chromaFull(const RgbToYuvExactParams& p, int r, int g, int b,
                                           int& u, int& v)
{
    if constexpr (!kDeepSample<E>)
    {
        u = (p.ru * r + p.gu * g + p.bu * b + (256 << 14) + (1 << 8)) >> 9;
        v = (p.rv * r + p.gv * g + p.bv * b + (256 << 14) + (1 << 8)) >> 9;
    }
    else
    {
        // rgb48ToUV_c: int32 coefficients times unsigned samples, i.e.
        // unsigned arithmetic; the true sum is always in [0, 2^31).
        u = int(((unsigned(p.ru) * unsigned(r) + unsigned(p.gu) * unsigned(g) +
                  unsigned(p.bu) * unsigned(b) + (0x10001u << 14)) >> 15) & 0xFFFFu);
        v = int(((unsigned(p.rv) * unsigned(r) + unsigned(p.gv) * unsigned(g) +
                  unsigned(p.bv) * unsigned(b) + (0x10001u << 14)) >> 15) & 0xFFFFu);
    }
}

template <typename E>
__device__ __forceinline__ void chromaHalf(const RgbToYuvExactParams& p, int r0, int g0, int b0,
                                           int r1, int g1, int b1, int& u, int& v)
{
    if constexpr (!kDeepSample<E>)
    {
        const int r = r0 + r1, g = g0 + g1, b = b0 + b1;
        u = (p.ru * r + p.gu * g + p.bu * b + (256 << 15) + (1 << 9)) >> 10;
        v = (p.rv * r + p.gv * g + p.bv * b + (256 << 15) + (1 << 9)) >> 10;
    }
    else
    {
        chromaFull<E>(p, (r0 + r1 + 1) >> 1, (g0 + g1 + 1) >> 1, (b0 + b1 + 1) >> 1, u, v);
    }
}

// Identity hscale (unscaled chroma) to 15 bits.
template <typename E>
__device__ __forceinline__ int hscaleIdentity(int s)
{
    if constexpr (!kDeepSample<E>)
        return min(2 * s, 32767); // hscale14to15, (s * 16384) >> 13
    else
        return s >> 1;            // hscale16to15, (s * 16384) >> 15
}

__device__ __forceinline__ int chromaRange(const RgbToYuvExactParams& p, int c15)
{
    return p.rangeConvert ? min((c15 * p.chrRangeCoeff + p.chrRangeOffset) >> 14, 32767) : c15;
}

// yuv2plane1 for luma and unscaled chroma.
template <int Bits>
__device__ __forceinline__ int plane1(int v15)
{
    if constexpr (Bits == 8)
        return clampInt((v15 + 64) >> 7, 0, 255);
    else
        return clampInt((v15 + 16) >> 5, 0, 1023);
}

//------------------------------------------------------------------------------
// Loads: kPx consecutive pixels of one row
//------------------------------------------------------------------------------
struct Px
{
    int r[kPx], g[kPx], b[kPx];
};

// Load NB bytes (NB % 4 == 0) with the widest access the address allows.
template <int NB>
__device__ __forceinline__ void loadWords(const uint8_t* p, uint32_t (&w)[NB / 4])
{
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    if constexpr (NB % 16 == 0)
    {
        if ((a & 15) == 0)
        {
#pragma unroll
            for (int i = 0; i < NB / 16; i++)
            {
                const uint4 q = __ldg(reinterpret_cast<const uint4*>(p) + i);
                w[4 * i] = q.x; w[4 * i + 1] = q.y; w[4 * i + 2] = q.z; w[4 * i + 3] = q.w;
            }
            return;
        }
    }
    if constexpr (NB % 8 == 0)
    {
        if ((a & 7) == 0)
        {
#pragma unroll
            for (int i = 0; i < NB / 8; i++)
            {
                const uint2 q = __ldg(reinterpret_cast<const uint2*>(p) + i);
                w[2 * i] = q.x; w[2 * i + 1] = q.y;
            }
            return;
        }
    }
    if ((a & 3) == 0)
    {
#pragma unroll
        for (int i = 0; i < NB / 4; i++)
            w[i] = __ldg(reinterpret_cast<const uint32_t*>(p) + i);
        return;
    }
#pragma unroll
    for (int i = 0; i < NB / 4; i++)
        w[i] = uint32_t(__ldg(p + 4 * i)) | (uint32_t(__ldg(p + 4 * i + 1)) << 8) |
               (uint32_t(__ldg(p + 4 * i + 2)) << 16) | (uint32_t(__ldg(p + 4 * i + 3)) << 24);
}

template <typename E>
__device__ __forceinline__ int element(const uint32_t* w, int i)
{
    if constexpr (sizeof(E) == 1)
        return int((w[i >> 2] >> (8 * (i & 3))) & 0xFFu);
    else
        return int((w[i >> 1] >> (16 * (i & 1))) & 0xFFFFu);
}

// PS = elements per pixel for interleaved (HWC) input, or 1 for planar
// (CHW) input whose channels are channelStride elements apart.
template <typename E, int PS>
__device__ __forceinline__ void loadPxInteger(const RgbToYuvExactSource& s, int x0, int y, int n,
                                              Px& px)
{
    const E* row = static_cast<const E*>(s.data) + int64_t(y) * s.rowStride;
    if (n == kPx)
    {
        if constexpr (PS == 1)
        {
            constexpr int NB = kPx * int(sizeof(E));
            uint32_t w[NB / 4];
            loadWords<NB>(reinterpret_cast<const uint8_t*>(row + x0), w);
#pragma unroll
            for (int i = 0; i < kPx; i++) px.r[i] = element<E>(w, i);
            loadWords<NB>(reinterpret_cast<const uint8_t*>(row + s.channelStride + x0), w);
#pragma unroll
            for (int i = 0; i < kPx; i++) px.g[i] = element<E>(w, i);
            loadWords<NB>(reinterpret_cast<const uint8_t*>(row + 2 * s.channelStride + x0), w);
#pragma unroll
            for (int i = 0; i < kPx; i++) px.b[i] = element<E>(w, i);
        }
        else
        {
            constexpr int NB = kPx * PS * int(sizeof(E));
            uint32_t w[NB / 4];
            loadWords<NB>(reinterpret_cast<const uint8_t*>(row + int64_t(x0) * PS), w);
#pragma unroll
            for (int i = 0; i < kPx; i++)
            {
                px.r[i] = element<E>(w, i * PS + 0);
                px.g[i] = element<E>(w, i * PS + 1);
                px.b[i] = element<E>(w, i * PS + 2);
            }
        }
        return;
    }
    const int64_t cs = (PS == 1) ? s.channelStride : 1;
#pragma unroll
    for (int i = 0; i < kPx; i++)
    {
        if (i < n)
        {
            const E* q = row + int64_t(x0 + i) * PS;
            px.r[i] = int(q[0]);
            px.g[i] = int(q[cs]);
            px.b[i] = int(q[2 * cs]);
        }
        else
        {
            px.r[i] = px.g[i] = px.b[i] = 0;
        }
    }
}

// One pixel, any layout (odd-width slow path, and every float source).
template <typename E>
__device__ __forceinline__ void loadPixel(const RgbToYuvExactSource& s, int x, int y,
                                          int& r, int& g, int& b)
{
    const int64_t i = int64_t(y) * s.rowStride + int64_t(x) * s.pixelStride;
    if constexpr (ElementTraits<E>::kFloat)
    {
        r = floatSample<E>(s.data, i, s.element);
        g = floatSample<E>(s.data, i + s.channelStride, s.element);
        b = floatSample<E>(s.data, i + 2 * s.channelStride, s.element);
    }
    else
    {
        const E* q = static_cast<const E*>(s.data) + i;
        r = int(q[0]);
        g = int(q[s.channelStride]);
        b = int(q[2 * s.channelStride]);
    }
}

// kPx consecutive pixels of row y. A float source reads through the
// strides, whatever PS is.
template <typename E, int PS>
__device__ __forceinline__ void loadPx(const RgbToYuvExactSource& s, int x0, int y, int n, Px& px)
{
    if constexpr (ElementTraits<E>::kFloat)
    {
#pragma unroll
        for (int i = 0; i < kPx; i++)
        {
            if (i < n)
                loadPixel<E>(s, x0 + i, y, px.r[i], px.g[i], px.b[i]);
            else
                px.r[i] = px.g[i] = px.b[i] = 0;
        }
    }
    else
    {
        loadPxInteger<E, PS>(s, x0, y, n, px);
    }
}

//------------------------------------------------------------------------------
// Stores: n output samples of one row
//------------------------------------------------------------------------------
template <int NB>
__device__ __forceinline__ void storeWords(uint8_t* p, const uint32_t (&w)[NB / 4])
{
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    if constexpr (NB % 16 == 0)
    {
        if ((a & 15) == 0)
        {
#pragma unroll
            for (int i = 0; i < NB / 16; i++)
                reinterpret_cast<uint4*>(p)[i] = make_uint4(w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]);
            return;
        }
    }
    if constexpr (NB % 8 == 0)
    {
        if ((a & 7) == 0)
        {
#pragma unroll
            for (int i = 0; i < NB / 8; i++)
                reinterpret_cast<uint2*>(p)[i] = make_uint2(w[2 * i], w[2 * i + 1]);
            return;
        }
    }
    if ((a & 3) == 0)
    {
#pragma unroll
        for (int i = 0; i < NB / 4; i++)
            reinterpret_cast<uint32_t*>(p)[i] = w[i];
        return;
    }
#pragma unroll
    for (int i = 0; i < NB; i++)
        p[i] = uint8_t(w[i >> 2] >> (8 * (i & 3)));
}

// Store N samples (8- or 16-bit) at p; the first n are valid.
template <int Bits, int N>
__device__ __forceinline__ void storeSamples(uint8_t* p, const int (&v)[N], int n)
{
    if constexpr (Bits == 8)
    {
        if constexpr (N % 4 == 0)
        {
            if (n == N)
            {
                uint32_t w[N / 4];
#pragma unroll
                for (int i = 0; i < N / 4; i++)
                    w[i] = uint32_t(v[4 * i]) | (uint32_t(v[4 * i + 1]) << 8) |
                           (uint32_t(v[4 * i + 2]) << 16) | (uint32_t(v[4 * i + 3]) << 24);
                storeWords<N>(p, w);
                return;
            }
        }
#pragma unroll
        for (int i = 0; i < N; i++)
            if (i < n) p[i] = uint8_t(v[i]);
    }
    else
    {
        if constexpr (N % 2 == 0)
        {
            if (n == N)
            {
                uint32_t w[N / 2];
#pragma unroll
                for (int i = 0; i < N / 2; i++)
                    w[i] = uint32_t(v[2 * i]) | (uint32_t(v[2 * i + 1]) << 16);
                storeWords<2 * N>(p, w);
                return;
            }
        }
#pragma unroll
        for (int i = 0; i < N; i++)
            if (i < n) reinterpret_cast<uint16_t*>(p)[i] = uint16_t(v[i]);
    }
}

template <ExactYuvTarget T>
__device__ __forceinline__ uint8_t* planeRow(const RgbToYuvExactDest& d, int plane, int y, int x)
{
    constexpr int bytes = TargetTraits<T>::kBits > 8 ? 2 : 1;
    return static_cast<uint8_t*>(d.plane[plane]) + int64_t(y) * d.pitch[plane] + int64_t(x) * bytes;
}

template <ExactYuvTarget T, int N>
__device__ __forceinline__ void storeChroma(const RgbToYuvExactDest& d, int cy, int cx,
                                            const int (&u)[N], const int (&v)[N], int n)
{
    using TT = TargetTraits<T>;
    if constexpr (TT::kSemiPlanar)
    {
        int uv[2 * N];
#pragma unroll
        for (int i = 0; i < N; i++)
        {
            uv[2 * i] = u[i] << TT::kMsbShift;
            uv[2 * i + 1] = v[i] << TT::kMsbShift;
        }
        storeSamples<TT::kBits, 2 * N>(planeRow<T>(d, 1, cy, 2 * cx), uv, 2 * n);
    }
    else
    {
        storeSamples<TT::kBits, N>(planeRow<T>(d, 1, cy, cx), u, n);
        storeSamples<TT::kBits, N>(planeRow<T>(d, 2, cy, cx), v, n);
    }
}

template <ExactYuvTarget T, typename E>
__device__ __forceinline__ void storeLuma(const RgbToYuvExactParams& p, const RgbToYuvExactDest& d,
                                          const Px& px, int y, int x0, int n)
{
    using TT = TargetTraits<T>;
    int out[kPx];
#pragma unroll
    for (int i = 0; i < kPx; i++)
        out[i] = plane1<TT::kBits>(lumaY15<E>(p, px.r[i], px.g[i], px.b[i])) << TT::kMsbShift;
    storeSamples<TT::kBits, kPx>(planeRow<T>(d, 0, y, x0), out, n);
}

//------------------------------------------------------------------------------
// 4:2:0 vertical chroma filter, one tap / the final rounding. The
// accumulator starts at vfilterInit.
//------------------------------------------------------------------------------
template <int Bits>
__device__ __forceinline__ int vfilterInit(bool mmx, int taps)
{
    // yuv2yuvX: ((filterSize - 1) * 8 + dither 64) >> 4, in 16-bit lanes.
    // C (yuv2planeX_8 / yuv2nv12cX / yuv2planeX_10 / yuv2p01xcX): the
    // rounding term, 64 << 12 at 8 bits and 1 << 16 at 10.
    return mmx ? (((taps - 1) * 8 + 64) >> 4) : (Bits == 8 ? (64 << 12) : (1 << 16));
}

__device__ __forceinline__ void vfilterTap(bool mmx, int c, int u, int v, int& au, int& av)
{
    if (mmx)
    {
        au += (c * u) >> 16; // pmulhw
        av += (c * v) >> 16;
    }
    else
    {
        au += u * c;
        av += v * c;
    }
}

template <int Bits>
__device__ __forceinline__ int vfilterOut(bool mmx, int acc)
{
    if (mmx)
        return clampInt(acc >> 3, 0, 255); // psraw 3 + packuswb
    if constexpr (Bits == 8)
        return clampInt(acc >> 19, 0, 255);
    else
        return clampInt(acc >> 17, 0, 1023);
}

//------------------------------------------------------------------------------
// 4:2:0, even width: one thread = 8 luma columns x 2 luma rows + 4 chroma
// samples of one chroma row.
//------------------------------------------------------------------------------
template <typename E, int PS, ExactYuvTarget T>
__global__ void __launch_bounds__(128)
Rgb2Yuv420EvenKernel(RgbToYuvExactParams p, RgbToYuvExactSource s, RgbToYuvExactDest d,
                     int width, int height)
{
    using TT = TargetTraits<T>;
    constexpr int kC = kPx / 2;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * kPx;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int chrH = (height + 1) >> 1;
    if (x0 >= width || k >= chrH)
        return;
    const int n = min(kPx, width - x0); // even: the width is even

    const int taps = p.vChrTaps;
    const int pos = p.vChrPos[k];
    const int16_t* coef = p.vChrCoef + int64_t(k) * taps;
    // yuv2yuvX only while dstY < dstH - 2; the last chroma row is C.
    const bool mmx = TT::kMmxVFilter && taps > 1 && 2 * k < height - 2;

    int accU[kC], accV[kC];
    const int acc0 = vfilterInit<TT::kBits>(mmx, taps);
#pragma unroll
    for (int i = 0; i < kC; i++)
        accU[i] = accV[i] = acc0;

    // Visit every source row this thread needs once: its chroma taps and its
    // two luma rows (the taps normally include them).
    const int ly0 = 2 * k;
    const int ly1 = min(2 * k + 1, height - 1);
    const int rBeg = min(pos, ly0);
    const int rEnd = max(pos + taps - 1, ly1);
    for (int r = rBeg; r <= rEnd; r++)
    {
        Px px;
        loadPx<E, PS>(s, x0, clampInt(r, 0, height - 1), n, px);
        if (r == ly0 || r == 2 * k + 1)
            if (r < height)
                storeLuma<T, E>(p, d, px, r, x0, n);
        const int j = r - pos;
        if (j >= 0 && j < taps)
        {
            const int c = coef[j];
#pragma unroll
            for (int i = 0; i < kC; i++)
            {
                int u, v;
                chromaHalf<E>(p, px.r[2 * i], px.g[2 * i], px.b[2 * i],
                              px.r[2 * i + 1], px.g[2 * i + 1], px.b[2 * i + 1], u, v);
                u = chromaRange(p, hscaleIdentity<E>(u));
                v = chromaRange(p, hscaleIdentity<E>(v));
                vfilterTap(mmx, c, u, v, accU[i], accV[i]);
            }
        }
    }

    int outU[kC], outV[kC];
#pragma unroll
    for (int i = 0; i < kC; i++)
    {
        outU[i] = vfilterOut<TT::kBits>(mmx, accU[i]);
        outV[i] = vfilterOut<TT::kBits>(mmx, accV[i]);
    }
    storeChroma<T, kC>(d, k, x0 / 2, outU, outV, n / 2);
}

//------------------------------------------------------------------------------
// 4:2:0 variant: two chroma rows (4 luma rows) per thread, so each source
// row is read 1.5x instead of 2x on average.
//------------------------------------------------------------------------------
template <typename E, int PS, ExactYuvTarget T>
__global__ void __launch_bounds__(128)
Rgb2Yuv420Even2Kernel(RgbToYuvExactParams p, RgbToYuvExactSource s, RgbToYuvExactDest d,
                      int width, int height)
{
    using TT = TargetTraits<T>;
    constexpr int kC = kPx / 2;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * kPx;
    const int k0 = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const int chrH = (height + 1) >> 1;
    if (x0 >= width || k0 >= chrH)
        return;
    const int n = min(kPx, width - x0);
    const int rows = min(2, chrH - k0); // chroma rows this thread owns

    const int taps = p.vChrTaps;
    int pos[2];
    bool mmx[2];
    const int16_t* coef[2];
    int accU[2][kC], accV[2][kC];
#pragma unroll
    for (int c = 0; c < 2; c++)
    {
        const int k = min(k0 + c, chrH - 1);
        pos[c] = p.vChrPos[k];
        coef[c] = p.vChrCoef + int64_t(k) * taps;
        mmx[c] = TT::kMmxVFilter && taps > 1 && 2 * k < height - 2;
        const int acc0 = vfilterInit<TT::kBits>(mmx[c], taps);
#pragma unroll
        for (int i = 0; i < kC; i++)
            accU[c][i] = accV[c][i] = acc0;
    }

    const int lyEnd = min(2 * (k0 + rows) - 1, height - 1); // last luma row owned
    const int rBeg = min(pos[0], 2 * k0);
    const int rEnd = max(pos[rows - 1] + taps - 1, lyEnd);
    for (int r = rBeg; r <= rEnd; r++)
    {
        Px px;
        loadPx<E, PS>(s, x0, clampInt(r, 0, height - 1), n, px);
        if (r >= 2 * k0 && r <= lyEnd)
            storeLuma<T, E>(p, d, px, r, x0, n);
        const int j0 = r - pos[0];
        const int j1 = r - pos[1];
        const bool in0 = j0 >= 0 && j0 < taps;
        const bool in1 = rows > 1 && j1 >= 0 && j1 < taps;
        if (!in0 && !in1)
            continue;
        const int c0 = in0 ? coef[0][j0] : 0;
        const int c1 = in1 ? coef[1][j1] : 0;
#pragma unroll
        for (int i = 0; i < kC; i++)
        {
            int u, v;
            chromaHalf<E>(p, px.r[2 * i], px.g[2 * i], px.b[2 * i],
                          px.r[2 * i + 1], px.g[2 * i + 1], px.b[2 * i + 1], u, v);
            u = chromaRange(p, hscaleIdentity<E>(u));
            v = chromaRange(p, hscaleIdentity<E>(v));
            if (in0)
                vfilterTap(mmx[0], c0, u, v, accU[0][i], accV[0][i]);
            if (in1)
                vfilterTap(mmx[1], c1, u, v, accU[1][i], accV[1][i]);
        }
    }

#pragma unroll
    for (int c = 0; c < 2; c++)
    {
        if (c >= rows)
            break;
        int outU[kC], outV[kC];
#pragma unroll
        for (int i = 0; i < kC; i++)
        {
            outU[i] = vfilterOut<TT::kBits>(mmx[c], accU[c][i]);
            outV[i] = vfilterOut<TT::kBits>(mmx[c], accV[c][i]);
        }
        storeChroma<T, kC>(d, k0 + c, x0 / 2, outU, outV, n / 2);
    }
}

//------------------------------------------------------------------------------
// 4:4:4 (any width) and 4:2:2 (even width): one thread = 8 pixels of a row.
//------------------------------------------------------------------------------
template <typename E, int PS, ExactYuvTarget T>
__global__ void __launch_bounds__(128)
Rgb2YuvRowKernel(RgbToYuvExactParams p, RgbToYuvExactSource s, RgbToYuvExactDest d,
                 int width, int height)
{
    using TT = TargetTraits<T>;
    constexpr int kC = kPx >> TT::kShiftX;
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * kPx;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x0 >= width || y >= height)
        return;
    const int n = min(kPx, width - x0);

    Px px;
    loadPx<E, PS>(s, x0, y, n, px);
    storeLuma<T, E>(p, d, px, y, x0, n);

    int outU[kC], outV[kC];
#pragma unroll
    for (int i = 0; i < kC; i++)
    {
        int u, v;
        if constexpr (TT::kShiftX)
            chromaHalf<E>(p, px.r[2 * i], px.g[2 * i], px.b[2 * i],
                          px.r[2 * i + 1], px.g[2 * i + 1], px.b[2 * i + 1], u, v);
        else
            chromaFull<E>(p, px.r[i], px.g[i], px.b[i], u, v);
        outU[i] = plane1<TT::kBits>(chromaRange(p, hscaleIdentity<E>(u)));
        outV[i] = plane1<TT::kBits>(chromaRange(p, hscaleIdentity<E>(v)));
    }
    storeChroma<T, kC>(d, y, x0 >> TT::kShiftX, outU, outV, n >> TT::kShiftX);
}

//------------------------------------------------------------------------------
// Odd-width 4:2:x (rare; correctness over speed). swscale keeps chroma at
// full resolution there and runs a real bilinear hscale.
//------------------------------------------------------------------------------
template <typename E, ExactYuvTarget T>
__global__ void __launch_bounds__(128)
Rgb2YuvLumaKernel(RgbToYuvExactParams p, RgbToYuvExactSource s, RgbToYuvExactDest d,
                  int width, int height)
{
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * kPx;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x0 >= width || y >= height)
        return;
    const int n = min(kPx, width - x0);
    Px px;
#pragma unroll
    for (int i = 0; i < kPx; i++)
    {
        if (i < n)
            loadPixel<E>(s, x0 + i, y, px.r[i], px.g[i], px.b[i]);
        else
            px.r[i] = px.g[i] = px.b[i] = 0;
    }
    storeLuma<T, E>(p, d, px, y, x0, n);
}

// hscale of the full-resolution chroma buffer at one output column, then the
// range pass. 14-bit input: hscale14to15 (pmaddwd, >> 13, packssdw).
// 16-bit input: hscale16to15 biases samples by -0x8000 and adds back
// 0x8000 * 16384 regardless of the taps' actual sum, then >> 15, packssdw.
template <typename E>
__device__ __forceinline__ void hscaleChroma(const RgbToYuvExactParams& p,
                                             const RgbToYuvExactSource& s, int y, int cx,
                                             int width, int& u15, int& v15)
{
    const int taps = p.hChrTaps;
    const int pos = p.hChrPos[cx];
    const int16_t* coef = p.hChrCoef + int64_t(cx) * taps;
    int su = 0, sv = 0;
    for (int t = 0; t < taps; t++)
    {
        const int c = coef[t];
        if (c == 0)
            continue; // taps past the edge are zero; skip their loads
        int r, g, b, u, v;
        loadPixel<E>(s, min(pos + t, width - 1), y, r, g, b);
        chromaFull<E>(p, r, g, b, u, v);
        if constexpr (!kDeepSample<E>)
        {
            su += u * c;
            sv += v * c;
        }
        else
        {
            su += (u - 0x8000) * c;
            sv += (v - 0x8000) * c;
        }
    }
    if constexpr (!kDeepSample<E>)
    {
        u15 = clampInt(su >> 13, -32768, 32767);
        v15 = clampInt(sv >> 13, -32768, 32767);
    }
    else
    {
        u15 = clampInt(int(unsigned(su) + 0x20000000u) >> 15, -32768, 32767);
        v15 = clampInt(int(unsigned(sv) + 0x20000000u) >> 15, -32768, 32767);
    }
    u15 = chromaRange(p, u15);
    v15 = chromaRange(p, v15);
}

template <typename E, ExactYuvTarget T>
__global__ void __launch_bounds__(128)
Rgb2YuvOddChromaKernel(RgbToYuvExactParams p, RgbToYuvExactSource s, RgbToYuvExactDest d,
                       int width, int height)
{
    using TT = TargetTraits<T>;
    const int chrW = (width + 1) >> 1;
    const int chrH = (height + TT::kShiftY) >> TT::kShiftY;
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= chrW || k >= chrH)
        return;

    int outU[1], outV[1];
    if constexpr (TT::kShiftY == 0)
    {
        int u, v;
        hscaleChroma<E>(p, s, k, cx, width, u, v);
        outU[0] = plane1<TT::kBits>(u);
        outV[0] = plane1<TT::kBits>(v);
    }
    else
    {
        const int taps = p.vChrTaps;
        const int pos = p.vChrPos[k];
        const int16_t* coef = p.vChrCoef + int64_t(k) * taps;
        const bool mmx = TT::kMmxVFilter && taps > 1 && 2 * k < height - 2;
        int au = vfilterInit<TT::kBits>(mmx, taps);
        int av = au;
        for (int j = 0; j < taps; j++)
        {
            const int c = coef[j];
            int u, v;
            hscaleChroma<E>(p, s, clampInt(pos + j, 0, height - 1), cx, width, u, v);
            vfilterTap(mmx, c, u, v, au, av);
        }
        outU[0] = vfilterOut<TT::kBits>(mmx, au);
        outV[0] = vfilterOut<TT::kBits>(mmx, av);
    }
    storeChroma<T, 1>(d, k, cx, outU, outV, 1);
}

//------------------------------------------------------------------------------
// Dispatch
//------------------------------------------------------------------------------
constexpr int kBlockX = 32;
constexpr int kBlockY = 4;

template <typename E, int PS, ExactYuvTarget T>
void launchFast(const RgbToYuvExactParams& p, const RgbToYuvExactSource& s,
                const RgbToYuvExactDest& d, int w, int h, cudaStream_t stream)
{
    using TT = TargetTraits<T>;
    const dim3 block(kBlockX, kBlockY);
    const int groups = (w + kPx - 1) / kPx;
    if constexpr (TT::kShiftY)
    {
        const int chrH = (h + 1) >> 1;
        // 16-bit sources move twice the bytes per tap row, so sharing each
        // row between two chroma rows pays off there (measured ~5-15% at
        // 4K); for 8-bit sources one row per thread is as fast or faster.
        if constexpr (kDeepSample<E> && !ElementTraits<E>::kFloat)
        {
            const dim3 grid((groups + kBlockX - 1) / kBlockX, ((chrH + 1) / 2 + kBlockY - 1) / kBlockY);
            Rgb2Yuv420Even2Kernel<E, PS, T><<<grid, block, 0, stream>>>(p, s, d, w, h);
        }
        else
        {
            const dim3 grid((groups + kBlockX - 1) / kBlockX, (chrH + kBlockY - 1) / kBlockY);
            Rgb2Yuv420EvenKernel<E, PS, T><<<grid, block, 0, stream>>>(p, s, d, w, h);
        }
    }
    else
    {
        const dim3 grid((groups + kBlockX - 1) / kBlockX, (h + kBlockY - 1) / kBlockY);
        Rgb2YuvRowKernel<E, PS, T><<<grid, block, 0, stream>>>(p, s, d, w, h);
    }
}

template <typename E, ExactYuvTarget T>
cudaError_t launchTarget(const RgbToYuvExactParams& p, const RgbToYuvExactSource& s,
                         const RgbToYuvExactDest& d, int w, int h, int channels,
                         cudaStream_t stream)
{
    using TT = TargetTraits<T>;
    // Interleaved input carries exactly the source's channels per pixel. A
    // float source is read through its strides, whatever they are.
    if (!ElementTraits<E>::kFloat && s.pixelStride != 1 &&
        (s.channelStride != 1 || s.pixelStride != channels))
        return cudaErrorInvalidValue;
    if (TT::kShiftX && (w & 1))
    {
        // Odd-width 4:2:x from a float frame is left to the torch path, to
        // keep the device code small (RgbToYuvExactConverter::reads).
        if constexpr (ElementTraits<E>::kFloat)
        {
            return cudaErrorInvalidValue;
        }
        else
        {
            if (!p.hChrPos || !p.hChrCoef || p.hChrTaps < 1 || p.hChrTaps > kExactMaxHTaps)
                return cudaErrorInvalidValue;
            const dim3 block(kBlockX, kBlockY);
            const dim3 lgrid(((w + kPx - 1) / kPx + kBlockX - 1) / kBlockX, (h + kBlockY - 1) / kBlockY);
            Rgb2YuvLumaKernel<E, T><<<lgrid, block, 0, stream>>>(p, s, d, w, h);
            const int chrW = (w + 1) >> 1;
            const int chrH = (h + TT::kShiftY) >> TT::kShiftY;
            const dim3 cgrid((chrW + kBlockX - 1) / kBlockX, (chrH + kBlockY - 1) / kBlockY);
            Rgb2YuvOddChromaKernel<E, T><<<cgrid, block, 0, stream>>>(p, s, d, w, h);
            return cudaGetLastError();
        }
    }

    if constexpr (ElementTraits<E>::kFloat)
        launchFast<E, 0, T>(p, s, d, w, h, stream); // any layout, see loadPx
    else if (s.pixelStride == 1)
        launchFast<E, 1, T>(p, s, d, w, h, stream);
    else if (s.channelStride == 1 && s.pixelStride == 3)
        launchFast<E, 3, T>(p, s, d, w, h, stream);
    else if (s.channelStride == 1 && s.pixelStride == 4)
        launchFast<E, 4, T>(p, s, d, w, h, stream);
    else
        return cudaErrorInvalidValue;
    return cudaGetLastError();
}

// A float frame is staged as 8-bit samples for an 8-bit target and as
// 16-bit ones for a 10-bit target, so only those pairs are instantiated.
template <typename E, ExactYuvTarget T>
cudaError_t launchCovered(const RgbToYuvExactParams& p, const RgbToYuvExactSource& s,
                          const RgbToYuvExactDest& d, int w, int h, int channels,
                          cudaStream_t stream)
{
    if constexpr (ElementTraits<E>::kFloat && (TargetTraits<T>::kBits > 8) != kDeepSample<E>)
        return cudaErrorInvalidValue;
    else
        return launchTarget<E, T>(p, s, d, w, h, channels, stream);
}

template <typename E>
cudaError_t launchSource(const RgbToYuvExactParams& p, ExactYuvTarget t,
                         const RgbToYuvExactSource& s, const RgbToYuvExactDest& d, int w,
                         int h, int channels, cudaStream_t stream)
{
    switch (t)
    {
    case ExactYuvTarget::Yuv420p:   return launchCovered<E, ExactYuvTarget::Yuv420p>(p, s, d, w, h, channels, stream);
    case ExactYuvTarget::Nv12:      return launchCovered<E, ExactYuvTarget::Nv12>(p, s, d, w, h, channels, stream);
    case ExactYuvTarget::Yuv422p:   return launchCovered<E, ExactYuvTarget::Yuv422p>(p, s, d, w, h, channels, stream);
    case ExactYuvTarget::Yuv444p:   return launchCovered<E, ExactYuvTarget::Yuv444p>(p, s, d, w, h, channels, stream);
    case ExactYuvTarget::Yuv420p10: return launchCovered<E, ExactYuvTarget::Yuv420p10>(p, s, d, w, h, channels, stream);
    case ExactYuvTarget::Yuv422p10: return launchCovered<E, ExactYuvTarget::Yuv422p10>(p, s, d, w, h, channels, stream);
    case ExactYuvTarget::Yuv444p10: return launchCovered<E, ExactYuvTarget::Yuv444p10>(p, s, d, w, h, channels, stream);
    case ExactYuvTarget::P010:      return launchCovered<E, ExactYuvTarget::P010>(p, s, d, w, h, channels, stream);
    }
    return cudaErrorInvalidValue;
}

// The source's samples are of type Sample (uint8_t / uint16_t); its
// elements are those samples or a float frame narrowed to them.
template <typename Sample>
cudaError_t launchElement(const RgbToYuvExactParams& p, ExactYuvTarget t,
                          const RgbToYuvExactSource& s, const RgbToYuvExactDest& d, int w,
                          int h, int channels, cudaStream_t stream)
{
    switch (s.element)
    {
    case ExactElement::Integer:
        return launchSource<Sample>(p, t, s, d, w, h, channels, stream);
    case ExactElement::Float32:
    case ExactElement::Float16:
    case ExactElement::BFloat16:
        return launchSource<FloatSamples<Sample>>(p, t, s, d, w, h, channels, stream);
    }
    return cudaErrorInvalidValue;
}

template <typename E>
__global__ void __launch_bounds__(256)
NarrowKernel(const void* src, ExactElement kind, int64_t count, uint16_t* dst)
{
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = uint16_t(floatSample<E>(src, i, kind));
}

template <typename Sample>
cudaError_t launchNarrow(ExactElement element, const void* src, int64_t count, uint16_t* dst,
                         cudaStream_t stream)
{
    const int64_t blocks = (count + 255) / 256;
    if (blocks > 0x7FFFFFFF)
        return cudaErrorInvalidValue;
    NarrowKernel<FloatSamples<Sample>><<<unsigned(blocks), 256, 0, stream>>>(src, element,
                                                                               count, dst);
    return cudaGetLastError();
}

} // namespace

cudaError_t RgbToYuvExact(const RgbToYuvExactParams& params, ExactRgbSource source,
                          ExactYuvTarget target, int width, int height,
                          const RgbToYuvExactSource& src, const RgbToYuvExactDest& dst,
                          cudaStream_t stream)
{
    if (width <= 0 || height <= 0 || !src.data || !dst.plane[0] || !dst.plane[1])
        return cudaErrorInvalidValue;
    const bool fourTwoZero = target == ExactYuvTarget::Yuv420p || target == ExactYuvTarget::Nv12 ||
                             target == ExactYuvTarget::Yuv420p10 || target == ExactYuvTarget::P010;
    if (fourTwoZero && (!params.vChrPos || !params.vChrCoef || params.vChrTaps < 1 ||
                        params.vChrTaps > kExactMaxVTaps))
        return cudaErrorInvalidValue;

    switch (source)
    {
    case ExactRgbSource::Rgb24:  return launchElement<uint8_t>(params, target, src, dst, width, height, 3, stream);
    case ExactRgbSource::Rgba:   return launchElement<uint8_t>(params, target, src, dst, width, height, 4, stream);
    case ExactRgbSource::Rgb48:  return launchElement<uint16_t>(params, target, src, dst, width, height, 3, stream);
    case ExactRgbSource::Rgba64: return launchElement<uint16_t>(params, target, src, dst, width, height, 4, stream);
    }
    return cudaErrorInvalidValue;
}

cudaError_t RgbToYuvExactNarrow(ExactElement element, int sampleBits, const void* src,
                                int64_t count, uint16_t* dst, cudaStream_t stream)
{
    if (element == ExactElement::Integer || count < 0 || (count && (!src || !dst)))
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    if (sampleBits == 8)
        return launchNarrow<uint8_t>(element, src, count, dst, stream);
    if (sampleBits == 16)
        return launchNarrow<uint16_t>(element, src, count, dst, stream);
    return cudaErrorInvalidValue;
}

} // namespace nelux::backends::cuda

#endif // NELUX_ENABLE_CUDA
