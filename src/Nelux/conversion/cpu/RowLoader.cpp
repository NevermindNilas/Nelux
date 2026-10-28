#include "RowLoader.hpp"
#include "RgbToYuvAvx2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#define NELUX_ROWLOAD_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#else
#define NELUX_ROWLOAD_X86 0
#endif

// As in RgbToYuvAvx2.cpp: AVX2 is not enabled globally, so the vector
// functions carry the target attribute outside MSVC. F16C (vcvtph2ps) ships
// with every AVX2 CPU but has its own CPUID bit, checked below.
#if defined(_MSC_VER) && !defined(__clang__)
#define NELUX_AVX2_F16C
#else
#define NELUX_AVX2_F16C __attribute__((target("avx2,f16c")))
#endif

namespace nelux
{
namespace conversion
{
namespace cpu
{

namespace
{

size_t sampleBytes(SampleType t)
{
    switch (t)
    {
    case SampleType::U8:   return 1;
    case SampleType::U16:
    case SampleType::F16:
    case SampleType::BF16: return 2;
    case SampleType::F32:  return 4;
    case SampleType::F64:  return 8;
    }
    return 1;
}

template <class T> inline T sampleAt(const uint8_t* p, int i)
{
    T v;
    std::memcpy(&v, p + size_t(i) * sizeof(T), sizeof(T));
    return v;
}

// torch: (x * 255.0f).clamp(0, 255).to(uint8). clamp keeps NaN and the cast
// turns it into 0.
inline uint8_t narrowFloat(float x)
{
    const float v = x * 255.0f;
    return v > 0.0f ? static_cast<uint8_t>(static_cast<int>(v < 255.0f ? v : 255.0f)) : 0;
}

// torch: (x * 65535.0f).round().clamp(0, 65535).to(uint16). round() goes
// half to even, as nearbyint does in the default rounding mode; clamp keeps
// NaN and the cast turns it into 0.
inline uint16_t narrowFloatU16(float x)
{
    const float v = std::nearbyint(x * 65535.0f);
    return v > 0.0f ? static_cast<uint16_t>(static_cast<int>(v < 65535.0f ? v : 65535.0f)) : 0;
}

inline void storeU16(uint8_t* dst, int i, uint16_t v)
{
    std::memcpy(dst + size_t(i) * 2, &v, 2);
}

inline float halfToFloat(uint16_t h)
{
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t bits;
    if (exponent == 0x1f)
        bits = sign | 0x7f800000u | (mantissa << 13);
    else if (exponent != 0)
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    else
    {
        // Zero or subnormal: mantissa * 2^-24 is exact in float.
        const float f = float(mantissa) * (1.0f / 16777216.0f);
        return sign ? -f : f;
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline float bf16ToFloat(uint16_t h)
{
    const uint32_t bits = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// n contiguous samples to n bytes.
void narrowScalar(SampleType t, const uint8_t* src, int n, uint8_t* dst)
{
    switch (t)
    {
    case SampleType::U8:
        std::memcpy(dst, src, size_t(n));
        break;
    case SampleType::U16:
        // x / 257 never leaves [0, 255], so torch's clamp is a no-op.
        for (int i = 0; i < n; i++)
            dst[i] = static_cast<uint8_t>(
                static_cast<int>(float(sampleAt<uint16_t>(src, i)) / 257.0f));
        break;
    case SampleType::F16:
        for (int i = 0; i < n; i++)
            dst[i] = narrowFloat(halfToFloat(sampleAt<uint16_t>(src, i)));
        break;
    case SampleType::BF16:
        for (int i = 0; i < n; i++)
            dst[i] = narrowFloat(bf16ToFloat(sampleAt<uint16_t>(src, i)));
        break;
    case SampleType::F32:
        for (int i = 0; i < n; i++)
            dst[i] = narrowFloat(sampleAt<float>(src, i));
        break;
    case SampleType::F64:
        for (int i = 0; i < n; i++)
            dst[i] = narrowFloat(static_cast<float>(sampleAt<double>(src, i)));
        break;
    }
}

void interleaveScalar(const uint8_t* const* planes, int channels, int n, uint8_t* dst)
{
    for (int x = 0; x < n; x++)
        for (int c = 0; c < channels; c++)
            dst[x * channels + c] = planes[c][x];
}

// n contiguous samples to n little-endian uint16 (2n bytes).
void narrowScalarU16(SampleType t, const uint8_t* src, int n, uint8_t* dst)
{
    switch (t)
    {
    case SampleType::U8:  // never staged 16-bit; widened to the full range
        for (int i = 0; i < n; i++)
            storeU16(dst, i, uint16_t(src[i] * 257));
        break;
    case SampleType::U16:
        std::memcpy(dst, src, size_t(n) * 2);
        break;
    case SampleType::F16:
        for (int i = 0; i < n; i++)
            storeU16(dst, i, narrowFloatU16(halfToFloat(sampleAt<uint16_t>(src, i))));
        break;
    case SampleType::BF16:
        for (int i = 0; i < n; i++)
            storeU16(dst, i, narrowFloatU16(bf16ToFloat(sampleAt<uint16_t>(src, i))));
        break;
    case SampleType::F32:
        for (int i = 0; i < n; i++)
            storeU16(dst, i, narrowFloatU16(sampleAt<float>(src, i)));
        break;
    case SampleType::F64:
        for (int i = 0; i < n; i++)
            storeU16(dst, i, narrowFloatU16(static_cast<float>(sampleAt<double>(src, i))));
        break;
    }
}

void interleaveScalarU16(const uint8_t* const* planes, int channels, int n, uint8_t* dst)
{
    for (int x = 0; x < n; x++)
        for (int c = 0; c < channels; c++)
            std::memcpy(dst + size_t(x * channels + c) * 2, planes[c] + size_t(x) * 2, 2);
}

#if NELUX_ROWLOAD_X86

bool cpuHasF16c()
{
    static const bool hasF16c = []
    {
#if defined(_MSC_VER)
        int regs[4];
        __cpuid(regs, 1);
        return (regs[2] & (1 << 29)) != 0;
#else
        unsigned a, b, c, d;
        return __get_cpuid(1, &a, &b, &c, &d) && (c & (1u << 29)) != 0;
#endif
    }();
    return hasF16c;
}

// 32 int32 lanes in [0, 255] (a..d, 8 each) to 32 bytes in order.
NELUX_AVX2_F16C inline __m256i packBytes(__m256i a, __m256i b, __m256i c, __m256i d)
{
    const __m256i ab = _mm256_packs_epi32(a, b);
    const __m256i cd = _mm256_packs_epi32(c, d);
    return _mm256_permutevar8x32_epi32(_mm256_packus_epi16(ab, cd),
                                       _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

NELUX_AVX2_F16C inline __m256i narrowFloat8(__m256 x)
{
    const __m256 v = _mm256_mul_ps(x, _mm256_set1_ps(255.0f));
    // max_ps returns its second operand when either one is NaN: NaN -> 0.
    return _mm256_cvttps_epi32(
        _mm256_min_ps(_mm256_max_ps(v, _mm256_setzero_ps()), _mm256_set1_ps(255.0f)));
}

// 8 float samples as float32, exactly as torch's .to(float32).
template <SampleType T> NELUX_AVX2_F16C inline __m256 loadFloat8(const uint8_t* p);

template <> NELUX_AVX2_F16C inline __m256 loadFloat8<SampleType::F16>(const uint8_t* p)
{
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

template <> NELUX_AVX2_F16C inline __m256 loadFloat8<SampleType::BF16>(const uint8_t* p)
{
    const __m256i x = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    return _mm256_castsi256_ps(_mm256_slli_epi32(x, 16));
}

template <> NELUX_AVX2_F16C inline __m256 loadFloat8<SampleType::F32>(const uint8_t* p)
{
    return _mm256_loadu_ps(reinterpret_cast<const float*>(p));
}

template <> NELUX_AVX2_F16C inline __m256 loadFloat8<SampleType::F64>(const uint8_t* p)
{
    const double* d = reinterpret_cast<const double*>(p);
    const __m128 lo = _mm256_cvtpd_ps(_mm256_loadu_pd(d));
    const __m128 hi = _mm256_cvtpd_ps(_mm256_loadu_pd(d + 4));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}

template <SampleType T> NELUX_AVX2_F16C inline __m256i narrow8(const uint8_t* p)
{
    return narrowFloat8(loadFloat8<T>(p));
}

template <> NELUX_AVX2_F16C inline __m256i narrow8<SampleType::U16>(const uint8_t* p)
{
    const __m256i x = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    return _mm256_cvttps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(x), _mm256_set1_ps(257.0f)));
}

template <SampleType T>
NELUX_AVX2_F16C void narrowAvx2(const uint8_t* src, int n, uint8_t* dst)
{
    constexpr size_t step = 8 * (T == SampleType::F32 ? 4 : T == SampleType::F64 ? 8 : 2);
    int i = 0;
    for (; i + 32 <= n; i += 32, src += 4 * step)
    {
        const __m256i bytes = packBytes(narrow8<T>(src), narrow8<T>(src + step),
                                        narrow8<T>(src + 2 * step), narrow8<T>(src + 3 * step));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), bytes);
    }
    _mm256_zeroupper();
    narrowScalar(T, src, n - i, dst + i);
}

void narrowSimd(SampleType t, const uint8_t* src, int n, uint8_t* dst)
{
    switch (t)
    {
    case SampleType::U8:   std::memcpy(dst, src, size_t(n)); return;
    case SampleType::U16:  narrowAvx2<SampleType::U16>(src, n, dst); return;
    case SampleType::F16:
        if (cpuHasF16c())
            narrowAvx2<SampleType::F16>(src, n, dst);
        else
            narrowScalar(t, src, n, dst);
        return;
    case SampleType::BF16: narrowAvx2<SampleType::BF16>(src, n, dst); return;
    case SampleType::F32:  narrowAvx2<SampleType::F32>(src, n, dst); return;
    case SampleType::F64:  narrowAvx2<SampleType::F64>(src, n, dst); return;
    }
}

// 8 float32 to 8 int32 in [0, 65535], as narrowFloatU16.
NELUX_AVX2_F16C inline __m256i narrowFloatU16x8(__m256 x)
{
    const __m256 v = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(65535.0f)),
                                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // max_ps returns its second operand when either one is NaN: NaN -> 0.
    return _mm256_cvttps_epi32(
        _mm256_min_ps(_mm256_max_ps(v, _mm256_setzero_ps()), _mm256_set1_ps(65535.0f)));
}

template <SampleType T>
NELUX_AVX2_F16C void narrowAvx2U16(const uint8_t* src, int n, uint8_t* dst)
{
    constexpr size_t step = 8 * (T == SampleType::F32 ? 4 : T == SampleType::F64 ? 8 : 2);
    int i = 0;
    for (; i + 16 <= n; i += 16, src += 2 * step)
    {
        const __m256i a = narrowFloatU16x8(loadFloat8<T>(src));
        const __m256i b = narrowFloatU16x8(loadFloat8<T>(src + step));
        // packus_epi32 works per 128-bit lane: a0-3 b0-3 a4-7 b4-7.
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 2 * size_t(i)),
                            _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xd8));
    }
    _mm256_zeroupper();
    narrowScalarU16(T, src, n - i, dst + 2 * size_t(i));
}

void narrowSimdU16(SampleType t, const uint8_t* src, int n, uint8_t* dst)
{
    switch (t)
    {
    case SampleType::U8:   narrowScalarU16(t, src, n, dst); return;
    case SampleType::U16:  std::memcpy(dst, src, size_t(n) * 2); return;
    case SampleType::F16:
        if (cpuHasF16c())
            narrowAvx2U16<SampleType::F16>(src, n, dst);
        else
            narrowScalarU16(t, src, n, dst);
        return;
    case SampleType::BF16: narrowAvx2U16<SampleType::BF16>(src, n, dst); return;
    case SampleType::F32:  narrowAvx2U16<SampleType::F32>(src, n, dst); return;
    case SampleType::F64:  narrowAvx2U16<SampleType::F64>(src, n, dst); return;
    }
}

// pshufb masks that spread 16 R, G and B bytes over three 16-byte blocks of
// packed RGB24: masks[block][channel][byte], -128 where the byte belongs to
// another channel.
struct Rgb24Masks
{
    alignas(16) int8_t m[3][3][16];
};

const Rgb24Masks& rgb24Masks()
{
    static const Rgb24Masks masks = []
    {
        Rgb24Masks t{};
        for (int block = 0; block < 3; block++)
            for (int j = 0; j < 16; j++)
            {
                const int i = 16 * block + j;
                for (int c = 0; c < 3; c++)
                    t.m[block][c][j] = (i % 3 == c) ? int8_t(i / 3) : int8_t(-128);
            }
        return t;
    }();
    return masks;
}

NELUX_AVX2_F16C void interleaveSimd(const uint8_t* const* planes, int channels, int n,
                                    uint8_t* dst)
{
    auto load = [](const uint8_t* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    };
    auto store = [](uint8_t* p, __m128i v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
    };
    int x = 0;
    if (channels == 3)
    {
        const Rgb24Masks& t = rgb24Masks();
        __m128i m[3][3];
        for (int block = 0; block < 3; block++)
            for (int c = 0; c < 3; c++)
                m[block][c] = load(reinterpret_cast<const uint8_t*>(t.m[block][c]));
        for (; x + 16 <= n; x += 16)
        {
            const __m128i r = load(planes[0] + x);
            const __m128i g = load(planes[1] + x);
            const __m128i b = load(planes[2] + x);
            for (int block = 0; block < 3; block++)
                store(dst + 3 * x + 16 * block,
                      _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, m[block][0]),
                                                _mm_shuffle_epi8(g, m[block][1])),
                                   _mm_shuffle_epi8(b, m[block][2])));
        }
    }
    else
    {
        for (; x + 16 <= n; x += 16)
        {
            const __m128i r = load(planes[0] + x);
            const __m128i g = load(planes[1] + x);
            const __m128i b = load(planes[2] + x);
            const __m128i a = load(planes[3] + x);
            const __m128i rgLo = _mm_unpacklo_epi8(r, g), rgHi = _mm_unpackhi_epi8(r, g);
            const __m128i baLo = _mm_unpacklo_epi8(b, a), baHi = _mm_unpackhi_epi8(b, a);
            store(dst + 4 * x, _mm_unpacklo_epi16(rgLo, baLo));
            store(dst + 4 * x + 16, _mm_unpackhi_epi16(rgLo, baLo));
            store(dst + 4 * x + 32, _mm_unpacklo_epi16(rgHi, baHi));
            store(dst + 4 * x + 48, _mm_unpackhi_epi16(rgHi, baHi));
        }
    }
    const uint8_t* rest[4];
    for (int c = 0; c < channels; c++)
        rest[c] = planes[c] + x;
    interleaveScalar(rest, channels, n - x, dst + size_t(x) * channels);
}

// As rgb24Masks, for 8 16-bit R, G and B samples over the three 16-byte
// blocks of packed RGB48.
const Rgb24Masks& rgb48Masks()
{
    static const Rgb24Masks masks = []
    {
        Rgb24Masks t{};
        for (int block = 0; block < 3; block++)
            for (int j = 0; j < 16; j++)
            {
                const int i = 16 * block + j;
                const int sample = i / 2;
                for (int c = 0; c < 3; c++)
                    t.m[block][c][j] = (sample % 3 == c) ? int8_t(2 * (sample / 3) + i % 2)
                                                         : int8_t(-128);
            }
        return t;
    }();
    return masks;
}

NELUX_AVX2_F16C void interleaveSimdU16(const uint8_t* const* planes, int channels, int n,
                                       uint8_t* dst)
{
    auto load = [](const uint8_t* p) {
        return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    };
    auto store = [](uint8_t* p, __m128i v) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
    };
    int x = 0;
    if (channels == 3)
    {
        const Rgb24Masks& t = rgb48Masks();
        __m128i m[3][3];
        for (int block = 0; block < 3; block++)
            for (int c = 0; c < 3; c++)
                m[block][c] = load(reinterpret_cast<const uint8_t*>(t.m[block][c]));
        for (; x + 8 <= n; x += 8)
        {
            const __m128i r = load(planes[0] + 2 * x);
            const __m128i g = load(planes[1] + 2 * x);
            const __m128i b = load(planes[2] + 2 * x);
            for (int block = 0; block < 3; block++)
                store(dst + 6 * x + 16 * block,
                      _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, m[block][0]),
                                                _mm_shuffle_epi8(g, m[block][1])),
                                   _mm_shuffle_epi8(b, m[block][2])));
        }
    }
    else
    {
        for (; x + 8 <= n; x += 8)
        {
            const __m128i r = load(planes[0] + 2 * x);
            const __m128i g = load(planes[1] + 2 * x);
            const __m128i b = load(planes[2] + 2 * x);
            const __m128i a = load(planes[3] + 2 * x);
            const __m128i rgLo = _mm_unpacklo_epi16(r, g), rgHi = _mm_unpackhi_epi16(r, g);
            const __m128i baLo = _mm_unpacklo_epi16(b, a), baHi = _mm_unpackhi_epi16(b, a);
            store(dst + 8 * x, _mm_unpacklo_epi32(rgLo, baLo));
            store(dst + 8 * x + 16, _mm_unpackhi_epi32(rgLo, baLo));
            store(dst + 8 * x + 32, _mm_unpacklo_epi32(rgHi, baHi));
            store(dst + 8 * x + 48, _mm_unpackhi_epi32(rgHi, baHi));
        }
    }
    const uint8_t* rest[4];
    for (int c = 0; c < channels; c++)
        rest[c] = planes[c] + 2 * x;
    interleaveScalarU16(rest, channels, n - x, dst + size_t(x) * channels * 2);
}

#endif // NELUX_ROWLOAD_X86

using NarrowFn = void (*)(SampleType, const uint8_t*, int, uint8_t*);
using InterleaveFn = void (*)(const uint8_t* const*, int, int, uint8_t*);

// outBytes is the size of a destination sample, 1 or 2; planes of the
// integer type of that size interleave as they are.
void loadRows(const RowSource& s, int width, int y0, int y1, uint8_t* dst, ptrdiff_t dstStride,
              int outBytes, NarrowFn narrow, InterleaveFn interleave)
{
    const size_t es = sampleBytes(s.type);
    const int channels = s.channels;
    const bool planar = s.pixelStride == 1;
    const bool asIs = s.type == (outBytes == 1 ? SampleType::U8 : SampleType::U16);
    // Other planar rows narrow a chunk of each plane, then interleave it.
    constexpr int kChunk = 256;
    alignas(32) uint8_t narrowed[4][2 * kChunk];

    for (int y = y0; y < y1; y++)
    {
        const uint8_t* row = s.data + size_t(y) * size_t(s.rowStride) * es;
        uint8_t* out = dst + ptrdiff_t(y - y0) * dstStride;
        if (!planar)
        {
            narrow(s.type, row, width * channels, out);
            continue;
        }
        const uint8_t* planes[4];
        for (int c = 0; c < channels; c++)
            planes[c] = row + size_t(c) * size_t(s.channelStride) * es;
        if (asIs)
        {
            interleave(planes, channels, width, out);
            continue;
        }
        for (int x0 = 0; x0 < width; x0 += kChunk)
        {
            const int n = std::min(kChunk, width - x0);
            const uint8_t* chunk[4];
            for (int c = 0; c < channels; c++)
            {
                narrow(s.type, planes[c] + size_t(x0) * es, n, narrowed[c]);
                chunk[c] = narrowed[c];
            }
            interleave(chunk, channels, n, out + size_t(x0) * channels * outBytes);
        }
    }
}

} // namespace

bool rowLoaderCovers(const RowSource& s)
{
    if (!s.data || (s.channels != 3 && s.channels != 4) || s.rowStride < 0 ||
        s.channelStride < 0)
        return false;
    return (s.pixelStride == s.channels && s.channelStride == 1) || s.pixelStride == 1;
}

bool isPackedU8(const RowSource& s, int width)
{
    return s.type == SampleType::U8 && s.channelStride == 1 && s.pixelStride == s.channels &&
           s.rowStride == int64_t(width) * s.channels;
}

void loadRowsU8(const RowSource& src, int width, int y0, int y1, uint8_t* dst,
                ptrdiff_t dstStride)
{
#if NELUX_ROWLOAD_X86
    if (simd::cpuHasAvx2())
    {
        loadRows(src, width, y0, y1, dst, dstStride, 1, narrowSimd, interleaveSimd);
        return;
    }
#endif
    loadRowsU8Reference(src, width, y0, y1, dst, dstStride);
}

void loadRowsU8Reference(const RowSource& src, int width, int y0, int y1, uint8_t* dst,
                         ptrdiff_t dstStride)
{
    loadRows(src, width, y0, y1, dst, dstStride, 1, narrowScalar, interleaveScalar);
}

bool isPackedU16(const RowSource& s, int width)
{
    return s.type == SampleType::U16 && s.channelStride == 1 && s.pixelStride == s.channels &&
           s.rowStride == int64_t(width) * s.channels;
}

void loadRowsU16(const RowSource& src, int width, int y0, int y1, uint8_t* dst,
                 ptrdiff_t dstStride)
{
#if NELUX_ROWLOAD_X86
    if (simd::cpuHasAvx2())
    {
        loadRows(src, width, y0, y1, dst, dstStride, 2, narrowSimdU16, interleaveSimdU16);
        return;
    }
#endif
    loadRowsU16Reference(src, width, y0, y1, dst, dstStride);
}

void loadRowsU16Reference(const RowSource& src, int width, int y0, int y1, uint8_t* dst,
                          ptrdiff_t dstStride)
{
    loadRows(src, width, y0, y1, dst, dstStride, 2, narrowScalarU16, interleaveScalarU16);
}

} // namespace cpu
} // namespace conversion
} // namespace nelux
