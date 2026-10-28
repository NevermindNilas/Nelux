#include "RgbToRgbAvx2.hpp"
#include "RgbToYuvAvx2.hpp" // cpuHasAvx2

extern "C"
{
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cstring>

// x86-64 only, like RgbToYuvAvx2.cpp: elsewhere swscale runs other code with
// other rounding, so no plan is made and every conversion stays on swscale.
#if defined(__x86_64__) || defined(_M_X64)
#define NELUX_RGB2RGB_X86 1
#include <immintrin.h>
#else
#define NELUX_RGB2RGB_X86 0
#endif

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

namespace
{

#if NELUX_RGB2RGB_X86

// ---------------------------------------------------------------------------
// Plan construction.
// ---------------------------------------------------------------------------

// roundToInt16 (yuv2rgb.c).
int32_t roundToInt16(int64_t f)
{
    const int r = int((f + (1 << 15)) >> 16);
    return r < -0x7FFF ? int16_t(0x8000) : r > 0x7FFF ? 0x7FFF : r;
}

// What swscale sets up for an RGB source and an RGB destination: both ranges
// forced limited (range_override_needed), SWS_CS_DEFAULT both ways, so
// fill_rgb2yuv_table takes its hard-coded BT.601 branch and
// ff_yuv2rgb_c_init_tables gets the default table with contrast and
// saturation 1 << 16, brightness 0.
void fillDeepCoefficients(RgbToRgbPlan& p)
{
    p.by = int32_t(0.114 * 219 / 255 * (1 << 15) + 0.5);
    p.bv = -int32_t(0.081 * 224 / 255 * (1 << 15) + 0.5);
    p.bu = int32_t(0.500 * 224 / 255 * (1 << 15) + 0.5);
    p.gy = int32_t(0.587 * 219 / 255 * (1 << 15) + 0.5);
    p.gv = -int32_t(0.419 * 224 / 255 * (1 << 15) + 0.5);
    p.gu = -int32_t(0.331 * 224 / 255 * (1 << 15) + 0.5);
    p.ry = int32_t(0.299 * 219 / 255 * (1 << 15) + 0.5);
    p.rv = int32_t(0.500 * 224 / 255 * (1 << 15) + 0.5);
    p.ru = -int32_t(0.169 * 224 / 255 * (1 << 15) + 0.5);

    const int* inv = sws_getCoefficients(SWS_CS_DEFAULT);
    const int64_t crv = inv[0], cbu = inv[1], cgu = -inv[2], cgv = -inv[3];
    const int64_t cy = (int64_t(1) << 16) * 255 / 219;
    const int64_t oy = int64_t(16) << 16;
    p.yCoeff = int16_t(roundToInt16(cy * (1 << 13)));
    p.yOffset = int16_t(roundToInt16(oy * (1 << 9)));
    p.v2r = int16_t(roundToInt16(crv * (1 << 13)));
    p.v2g = int16_t(roundToInt16(cgv * (1 << 13)));
    p.u2g = int16_t(roundToInt16(cgu * (1 << 13)));
    p.u2b = int16_t(roundToInt16(cbu * (1 << 13)));
}

// Destination byte order of the 8-bit packed layouts, as channel indices
// (0 R, 1 G, 2 B, -1 filler).
bool packedOrder(AVPixelFormat f, RgbToRgbPlan& p)
{
    struct Entry
    {
        AVPixelFormat fmt;
        int bytes;
        int8_t o[4];
    };
    static const Entry kTable[] = {
        {AV_PIX_FMT_BGR24, 3, {2, 1, 0, -1}}, {AV_PIX_FMT_RGBA, 4, {0, 1, 2, -1}},
        {AV_PIX_FMT_BGRA, 4, {2, 1, 0, -1}},  {AV_PIX_FMT_ARGB, 4, {-1, 0, 1, 2}},
        {AV_PIX_FMT_ABGR, 4, {-1, 2, 1, 0}},  {AV_PIX_FMT_RGB0, 4, {0, 1, 2, -1}},
        {AV_PIX_FMT_BGR0, 4, {2, 1, 0, -1}},  {AV_PIX_FMT_0RGB, 4, {-1, 0, 1, 2}},
        {AV_PIX_FMT_0BGR, 4, {-1, 2, 1, 0}},
    };
    for (const Entry& e : kTable)
        if (e.fmt == f)
        {
            p.dstBytes = e.bytes;
            std::memcpy(p.order, e.o, 4);
            return true;
        }
    return false;
}

// ---------------------------------------------------------------------------
// Scalar model.
// ---------------------------------------------------------------------------

inline int clampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// One pixel through rgb24ToY/rgb24ToUV (the AVX2 input functions' rounding),
// hscale14to15's doubling, and yuv2gbrp_full_X with a one-tap (4096)
// vertical filter. Writes G, B, R.
inline void deepPixel(const RgbToRgbPlan& p, int r, int g, int b, int& G, int& B, int& R)
{
    const int y14 = (p.ry * r + p.gy * g + p.by * b + 0x80100) >> 9;
    const int u14 = int16_t((p.ru * r + p.gu * g + p.bu * b + 0x400100) >> 9);
    const int v14 = int16_t((p.rv * r + p.gv * g + p.bv * b + 0x400100) >> 9);
    const int y15 = clampInt(2 * int16_t(y14), -32768, 32767);
    const int u15 = clampInt(2 * u14, -32768, 32767);
    const int v15 = clampInt(2 * v14, -32768, 32767);

    // (1 << 9) + s * 4096, >> 10: exactly 4 * s; chroma also loses 128 << 19.
    int Y = 4 * y15;
    const int U = 4 * u15 - 65536;
    const int V = 4 * v15 - 65536;
    Y -= p.yOffset;
    Y = int(unsigned(Y) * unsigned(p.yCoeff));
    Y += 1 << (p.sh - 1);
    int Rr = int(unsigned(Y) + unsigned(V) * unsigned(p.v2r));
    int Gg = int(unsigned(Y) + unsigned(V) * unsigned(p.v2g) + unsigned(U) * unsigned(p.u2g));
    int Bb = int(unsigned(Y) + unsigned(U) * unsigned(p.u2b));
    // The C clips all three when any leaves 30 bits; clipping an in-range
    // value changes nothing, so clipping each is the same.
    Rr = clampInt(Rr, 0, (1 << 30) - 1);
    Gg = clampInt(Gg, 0, (1 << 30) - 1);
    Bb = clampInt(Bb, 0, (1 << 30) - 1);
    R = Rr >> p.sh;
    G = Gg >> p.sh;
    B = Bb >> p.sh;
}

// The 19-bit round trip: hscale16to19 makes 32 * y14 (sh 9 for RGB input),
// the output stage takes >> 2 (rgba64_full_1) or (* 4096 - 2^30) >> 14 +
// 2^16 (gbrp16_full_X), both 8 * y14; then 16-bit output with int32 wrap.
inline void deep16Pixel(const RgbToRgbPlan& p, int r, int g, int b, int& R, int& G, int& B)
{
    const int y14 = (p.ry * r + p.gy * g + p.by * b + 0x80100) >> 9;
    const int u14 = int16_t((p.ru * r + p.gu * g + p.bu * b + 0x400100) >> 9);
    const int v14 = int16_t((p.rv * r + p.gv * g + p.bv * b + 0x400100) >> 9);
    const int Yv = std::min(32 * y14, (1 << 19) - 1) >> 2;
    const int U = (std::min(32 * u14, (1 << 19) - 1) - (128 << 11)) >> 2;
    const int V = (std::min(32 * v14, (1 << 19) - 1) - (128 << 11)) >> 2;
    unsigned Y = unsigned(Yv - p.yOffset) * unsigned(p.yCoeff) + unsigned((1 << 13) - (1 << 29));
    const unsigned Rr = Y + unsigned(V) * unsigned(p.v2r);
    const unsigned Gg = Y + unsigned(V) * unsigned(p.v2g) + unsigned(U) * unsigned(p.u2g);
    const unsigned Bb = Y + unsigned(U) * unsigned(p.u2b);
    R = clampInt((int(Rr) >> 14) + (1 << 15), 0, 65535);
    G = clampInt((int(Gg) >> 14) + (1 << 15), 0, 65535);
    B = clampInt((int(Bb) >> 14) + (1 << 15), 0, 65535);
}

inline void storeU16(uint8_t* row, int index, int v)
{
    row[2 * index] = uint8_t(v);
    row[2 * index + 1] = uint8_t(v >> 8);
}

inline void storeFloat(uint8_t* row, int index, int v)
{
    const float f = (1.0f / 65535.0f) * float(v);
    std::memcpy(row + 4 * index, &f, 4);
}

void rowScalar(const RgbToRgbPlan& p, const uint8_t* s, uint8_t* const d[4], int x0, int x1)
{
    switch (p.kind)
    {
    case RgbKind::Copy:
        std::memcpy(d[0] + 3 * x0, s + 3 * x0, size_t(x1 - x0) * 3);
        break;
    case RgbKind::Packed:
        for (int x = x0; x < x1; x++)
            for (int k = 0; k < p.dstBytes; k++)
                d[0][x * p.dstBytes + k] = p.order[k] < 0 ? 255 : s[3 * x + p.order[k]];
        break;
    case RgbKind::Planar8:
        for (int x = x0; x < x1; x++)
        {
            d[0][x] = s[3 * x + 1];
            d[1][x] = s[3 * x + 2];
            d[2][x] = s[3 * x];
            if (p.alphaPlane)
                d[3][x] = 255;
        }
        break;
    case RgbKind::PlanarDeep:
        for (int x = x0; x < x1; x++)
        {
            int G, B, R;
            deepPixel(p, s[3 * x], s[3 * x + 1], s[3 * x + 2], G, B, R);
            reinterpret_cast<uint16_t*>(d[0])[x] = uint16_t(G);
            reinterpret_cast<uint16_t*>(d[1])[x] = uint16_t(B);
            reinterpret_cast<uint16_t*>(d[2])[x] = uint16_t(R);
        }
        break;
    case RgbKind::Deep16:
        for (int x = x0; x < x1; x++)
        {
            int R, G, B;
            deep16Pixel(p, s[3 * x], s[3 * x + 1], s[3 * x + 2], R, G, B);
            if (p.swapRB)
                std::swap(R, B);
            switch (p.deepOut)
            {
            case Deep16Out::Planar:
                storeU16(d[0], x, G), storeU16(d[1], x, B), storeU16(d[2], x, R);
                break;
            case Deep16Out::PlanarFloat:
                storeFloat(d[0], x, G), storeFloat(d[1], x, B), storeFloat(d[2], x, R);
                break;
            case Deep16Out::Packed3:
                storeU16(d[0], 3 * x, R), storeU16(d[0], 3 * x + 1, G), storeU16(d[0], 3 * x + 2, B);
                break;
            case Deep16Out::Packed4:
                storeU16(d[0], 4 * x, R), storeU16(d[0], 4 * x + 1, G), storeU16(d[0], 4 * x + 2, B);
                storeU16(d[0], 4 * x + 3, 65535);
                break;
            }
        }
        break;
    }
}

inline void rowPointers(const RgbToRgbPlan& p, uint8_t* const dst[4], const int dstStride[4],
                        int y, uint8_t* d[4])
{
    const bool packed = p.kind == RgbKind::Copy || p.kind == RgbKind::Packed ||
                        (p.kind == RgbKind::Deep16 && (p.deepOut == Deep16Out::Packed3 ||
                                                       p.deepOut == Deep16Out::Packed4));
    const int planes = packed ? 1 : p.alphaPlane ? 4 : 3;
    for (int i = 0; i < 4; i++)
        d[i] = i < planes ? dst[i] + ptrdiff_t(dstStride[i]) * y : nullptr;
}

// ---------------------------------------------------------------------------
// AVX2 kernels. Every vector loop stops where its widest load or store would
// pass the row's last pixel; the scalar model finishes the row.
// ---------------------------------------------------------------------------

NELUX_AVX2 inline __m256i loadPair(const uint8_t* lo, const uint8_t* hi)
{
    return _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(lo))),
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(hi)), 1);
}

NELUX_AVX2 inline __m256i laneMask(const int8_t m[16])
{
    return _mm256_setr_epi8(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10],
                            m[11], m[12], m[13], m[14], m[15], m[0], m[1], m[2], m[3], m[4],
                            m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14],
                            m[15]);
}

// 3 -> 3 bytes: five pixels per lane (15 of 16 bytes), so 10 pixels a step;
// the 16th byte of each store is overwritten by the next one, or (last
// step) lies inside the row because the loop keeps one pixel of margin.
NELUX_AVX2 int packed3Avx2(const RgbToRgbPlan& p, const uint8_t* s, uint8_t* d, int w)
{
    int8_t m[16];
    for (int i = 0; i < 5; i++)
        for (int k = 0; k < 3; k++)
            m[3 * i + k] = int8_t(3 * i + p.order[k]);
    m[15] = 15;
    const __m256i mask = laneMask(m);
    int x = 0;
    for (; x + 11 <= w; x += 10)
    {
        const __m256i v = _mm256_shuffle_epi8(loadPair(s + 3 * x, s + 3 * x + 15), mask);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 3 * x), _mm256_castsi256_si128(v));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 3 * x + 15), _mm256_extracti128_si256(v, 1));
    }
    return x;
}

// 3 -> 4 bytes: four pixels per lane, eight a step (32 bytes out).
NELUX_AVX2 int packed4Avx2(const RgbToRgbPlan& p, const uint8_t* s, uint8_t* d, int w)
{
    int8_t m[16], f[16];
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < 4; k++)
        {
            m[4 * i + k] = p.order[k] < 0 ? int8_t(-1) : int8_t(3 * i + p.order[k]);
            f[4 * i + k] = p.order[k] < 0 ? int8_t(-1) : 0;
        }
    const __m256i mask = laneMask(m);
    const __m256i fill = laneMask(f);
    int x = 0;
    // Each lane reads 16 bytes for 12 used: keep 2 pixels of margin.
    for (; x + 10 <= w; x += 8)
    {
        const __m256i v = _mm256_shuffle_epi8(loadPair(s + 3 * x, s + 3 * x + 12), mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(d + 4 * x), _mm256_or_si256(v, fill));
    }
    return x;
}

// rgb24 -> G, B, R planes: 16 pixels a step from two 16-byte reads per lane
// (s and s + 8 for pixels 0-7, s + 24 and s + 32 for 8-15).
NELUX_AVX2 int planar8Avx2(const RgbToRgbPlan& p, const uint8_t* s, uint8_t* const d[4], int w)
{
    // First read (bytes 0-15): G of pixels 0-4 into bytes 0-4, B into 8-12;
    // R of pixels 0-5 into 0-5 of the second mask.
    static const int8_t gb0[16] = {1, 4, 7, 10, 13, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1};
    static const int8_t gb1[16] = {-1, -1, -1, -1, -1, 8, 11, 14, -1, -1, -1, -1, -1, 9, 12, 15};
    static const int8_t r0[16] = {0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    static const int8_t r1[16] = {-1, -1, -1, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1};
    const __m256i mGB0 = laneMask(gb0), mGB1 = laneMask(gb1);
    const __m256i mR0 = laneMask(r0), mR1 = laneMask(r1);
    const __m256i alpha = _mm256_set1_epi8(-1);
    int x = 0;
    // The second read of the high lane ends at byte 3 * x + 48: 16 pixels.
    for (; x + 16 <= w; x += 16)
    {
        const uint8_t* q = s + 3 * x;
        const __m256i a = loadPair(q, q + 24);
        const __m256i b = loadPair(q + 8, q + 32);
        // gb: per lane G0-7 in bytes 0-7, B0-7 in 8-15; r: R0-7 in 0-7.
        const __m256i gb = _mm256_or_si256(_mm256_shuffle_epi8(a, mGB0), _mm256_shuffle_epi8(b, mGB1));
        const __m256i r = _mm256_or_si256(_mm256_shuffle_epi8(a, mR0), _mm256_shuffle_epi8(b, mR1));
        // qwords: [G lo, B lo, G hi, B hi] -> [G lo, G hi, B lo, B hi].
        const __m256i gbq = _mm256_permute4x64_epi64(gb, 0xD8);
        const __m256i rq = _mm256_permute4x64_epi64(r, 0x08);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d[0] + x), _mm256_castsi256_si128(gbq));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d[1] + x), _mm256_extracti128_si256(gbq, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d[2] + x), _mm256_castsi256_si128(rq));
        if (p.alphaPlane)
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d[3] + x), _mm256_castsi256_si128(alpha));
    }
    return x;
}

NELUX_AVX2 inline __m256i dot(__m256i rg, __m256i b, __m256i cRG, __m256i cB)
{
    return _mm256_add_epi32(_mm256_madd_epi16(rg, cRG), _mm256_madd_epi16(b, cB));
}

NELUX_AVX2 inline __m256i wordPair(int32_t lo, int32_t hi)
{
    return _mm256_set1_epi32(int32_t((uint32_t(hi) << 16) | (uint32_t(lo) & 0xFFFF)));
}

// (dot + rnd) >> 9 as int16 (packssdw), then hscale14to15's saturating
// doubling: 16 pixels in natural order from groups A (0-3 | 8-11) and B.
NELUX_AVX2 inline __m256i s15(__m256i rgA, __m256i bA, __m256i rgB, __m256i bB, __m256i cRG,
                              __m256i cB, __m256i rnd)
{
    const __m256i a = _mm256_srai_epi32(_mm256_add_epi32(dot(rgA, bA, cRG, cB), rnd), 9);
    const __m256i b = _mm256_srai_epi32(_mm256_add_epi32(dot(rgB, bB, cRG, cB), rnd), 9);
    const __m256i v = _mm256_packs_epi32(a, b);
    return _mm256_adds_epi16(v, v);
}

// yuv2gbrp_full_X on 8 pixels (int32), returning the three shifted planes.
struct DeepConsts
{
    __m256i yOff, yCoeff, round, v2r, v2g, u2g, u2b, uvBias, zero, max30;
    __m128i shift;
};

NELUX_AVX2 inline void deep8(const DeepConsts& c, __m256i y, __m256i u, __m256i v, __m256i& G,
                             __m256i& B, __m256i& R)
{
    y = _mm256_slli_epi32(y, 2);
    u = _mm256_sub_epi32(_mm256_slli_epi32(u, 2), c.uvBias);
    v = _mm256_sub_epi32(_mm256_slli_epi32(v, 2), c.uvBias);
    y = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(y, c.yOff), c.yCoeff), c.round);
    R = _mm256_add_epi32(y, _mm256_mullo_epi32(v, c.v2r));
    G = _mm256_add_epi32(y, _mm256_add_epi32(_mm256_mullo_epi32(v, c.v2g), _mm256_mullo_epi32(u, c.u2g)));
    B = _mm256_add_epi32(y, _mm256_mullo_epi32(u, c.u2b));
    R = _mm256_sra_epi32(_mm256_min_epi32(_mm256_max_epi32(R, c.zero), c.max30), c.shift);
    G = _mm256_sra_epi32(_mm256_min_epi32(_mm256_max_epi32(G, c.zero), c.max30), c.shift);
    B = _mm256_sra_epi32(_mm256_min_epi32(_mm256_max_epi32(B, c.zero), c.max30), c.shift);
}

NELUX_AVX2 inline void store16x16(uint8_t* d, __m256i lo, __m256i hi)
{
    // packus works per lane: [lo0-3 hi0-3 | lo4-7 hi4-7] -> restore order.
    const __m256i v = _mm256_permute4x64_epi64(_mm256_packus_epi32(lo, hi), 0xD8);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d), v);
}

// The 16-bit / float tail on 8 pixels: (Y' + term) >> 14, + 2^15 (int32
// wrap, as the asm and the full_1 C do). packus or an explicit clamp
// supplies the final clip to 16 bits.
NELUX_AVX2 inline void deep16x8(const DeepConsts& c, __m256i y, __m256i u, __m256i v, __m256i& G,
                                __m256i& B, __m256i& R)
{
    y = _mm256_slli_epi32(y, 2);
    u = _mm256_sub_epi32(_mm256_slli_epi32(u, 2), c.uvBias);
    v = _mm256_sub_epi32(_mm256_slli_epi32(v, 2), c.uvBias);
    y = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_sub_epi32(y, c.yOff), c.yCoeff), c.round);
    const __m256i debias = _mm256_set1_epi32(1 << 15);
    const __m256i tg = _mm256_add_epi32(_mm256_mullo_epi32(v, c.v2g), _mm256_mullo_epi32(u, c.u2g));
    R = _mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(y, _mm256_mullo_epi32(v, c.v2r)), 14), debias);
    G = _mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(y, tg), 14), debias);
    B = _mm256_add_epi32(_mm256_srai_epi32(_mm256_add_epi32(y, _mm256_mullo_epi32(u, c.u2b)), 14), debias);
}

NELUX_AVX2 inline void storeFloat8(uint8_t* d, __m256i v, __m256i max16, __m256 scale)
{
    v = _mm256_min_epi32(_mm256_max_epi32(v, _mm256_setzero_si256()), max16);
    _mm256_storeu_ps(reinterpret_cast<float*>(d), _mm256_mul_ps(_mm256_cvtepi32_ps(v), scale));
}

// 8 pixels of 16-bit R, G, B (natural order) into packed rgb48 / rgba64.
NELUX_AVX2 inline void storePacked8(const RgbToRgbPlan& p, uint8_t* d, __m128i r, __m128i g,
                                    __m128i b, const __m128i* masks3)
{
    if (p.deepOut == Deep16Out::Packed4)
    {
        const __m128i a = _mm_set1_epi16(-1);
        const __m128i rgLo = _mm_unpacklo_epi16(r, g), rgHi = _mm_unpackhi_epi16(r, g);
        const __m128i baLo = _mm_unpacklo_epi16(b, a), baHi = _mm_unpackhi_epi16(b, a);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d), _mm_unpacklo_epi32(rgLo, baLo));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 16), _mm_unpackhi_epi32(rgLo, baLo));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 32), _mm_unpacklo_epi32(rgHi, baHi));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 48), _mm_unpackhi_epi32(rgHi, baHi));
        return;
    }
    for (int k = 0; k < 3; k++)
    {
        const __m128i v = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(r, masks3[3 * k]),
                                                    _mm_shuffle_epi8(g, masks3[3 * k + 1])),
                                       _mm_shuffle_epi8(b, masks3[3 * k + 2]));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 16 * k), v);
    }
}

// rgb24 -> planar 9..14-bit (k16 false) or the 16-bit / float layouts.
template <bool k16>
NELUX_AVX2 int deepAvx2(const RgbToRgbPlan& p, const uint8_t* s, uint8_t* const d[4], int w)
{
    // Pixels 0-3 | 8-11 (group A) and 4-7 | 12-15 (group B) as (R, G) and
    // (B, 0) word pairs, from 16-byte reads at s, s + 8 | s + 24, s + 32.
    static const int8_t rgA[16] = {0, -1, 1, -1, 3, -1, 4, -1, 6, -1, 7, -1, 9, -1, 10, -1};
    static const int8_t bA[16] = {2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1};
    static const int8_t rgB[16] = {4, -1, 5, -1, 7, -1, 8, -1, 10, -1, 11, -1, 13, -1, 14, -1};
    static const int8_t bB[16] = {6, -1, -1, -1, 9, -1, -1, -1, 12, -1, -1, -1, 15, -1, -1, -1};
    const __m256i mRgA = laneMask(rgA), mBA = laneMask(bA), mRgB = laneMask(rgB), mBB = laneMask(bB);
    const __m256i yRG = wordPair(p.ry, p.gy), yB = wordPair(p.by, 0);
    const __m256i uRG = wordPair(p.ru, p.gu), uB = wordPair(p.bu, 0);
    const __m256i vRG = wordPair(p.rv, p.gv), vB = wordPair(p.bv, 0);
    const __m256i rndY = _mm256_set1_epi32(0x80100), rndC = _mm256_set1_epi32(0x400100);
    DeepConsts c;
    c.yOff = _mm256_set1_epi32(p.yOffset);
    c.yCoeff = _mm256_set1_epi32(p.yCoeff);
    c.round = _mm256_set1_epi32(k16 ? (1 << 13) - (1 << 29) : 1 << (p.sh - 1));
    c.v2r = _mm256_set1_epi32(p.v2r);
    c.v2g = _mm256_set1_epi32(p.v2g);
    c.u2g = _mm256_set1_epi32(p.u2g);
    c.u2b = _mm256_set1_epi32(p.u2b);
    c.uvBias = _mm256_set1_epi32(65536);
    c.zero = _mm256_setzero_si256();
    c.max30 = _mm256_set1_epi32((1 << 30) - 1);
    c.shift = _mm_cvtsi32_si128(p.sh);
    const __m256i max16 = _mm256_set1_epi32(65535);
    const __m256 scale = _mm256_set1_ps(1.0f / 65535.0f);
    // rgb48: output block k, channel ch -> the bytes of that channel's words.
    __m128i masks3[9];
    for (int k = 0; k < 3; k++)
        for (int ch = 0; ch < 3; ch++)
        {
            alignas(16) int8_t m[16];
            for (int wd = 0; wd < 8; wd++)
            {
                const int gi = 8 * k + wd, px = gi / 3;
                const bool mine = gi % 3 == ch;
                m[2 * wd] = mine ? int8_t(2 * px) : int8_t(-1);
                m[2 * wd + 1] = mine ? int8_t(2 * px + 1) : int8_t(-1);
            }
            masks3[3 * k + ch] = _mm_load_si128(reinterpret_cast<const __m128i*>(m));
        }
    const int outBytes = p.deepOut == Deep16Out::Packed4 ? 8 : 6;
    int x = 0;
    for (; x + 16 <= w; x += 16)
    {
        const uint8_t* q = s + 3 * x;
        const __m256i a = loadPair(q, q + 24);
        const __m256i b = loadPair(q + 8, q + 32);
        const __m256i pRgA = _mm256_shuffle_epi8(a, mRgA), pBA = _mm256_shuffle_epi8(a, mBA);
        const __m256i pRgB = _mm256_shuffle_epi8(b, mRgB), pBB = _mm256_shuffle_epi8(b, mBB);
        // 16 int16 each, lane order [0-3 4-7 | 8-11 12-15] = natural.
        const __m256i y = s15(pRgA, pBA, pRgB, pBB, yRG, yB, rndY);
        const __m256i u = s15(pRgA, pBA, pRgB, pBB, uRG, uB, rndC);
        const __m256i v = s15(pRgA, pBA, pRgB, pBB, vRG, vB, rndC);
        const __m256i y0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(y));
        const __m256i u0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(u));
        const __m256i v0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
        const __m256i y1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(y, 1));
        const __m256i u1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(u, 1));
        const __m256i v1 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));
        __m256i g0, b0, r0, g1, b1, r1;
        if constexpr (!k16)
        {
            deep8(c, y0, u0, v0, g0, b0, r0);
            deep8(c, y1, u1, v1, g1, b1, r1);
            store16x16(d[0] + 2 * x, g0, g1);
            store16x16(d[1] + 2 * x, b0, b1);
            store16x16(d[2] + 2 * x, r0, r1);
        }
        else
        {
            deep16x8(c, y0, u0, v0, g0, b0, r0);
            deep16x8(c, y1, u1, v1, g1, b1, r1);
            if (p.swapRB)
            {
                std::swap(r0, b0);
                std::swap(r1, b1);
            }
            switch (p.deepOut)
            {
            case Deep16Out::Planar:
                store16x16(d[0] + 2 * x, g0, g1);
                store16x16(d[1] + 2 * x, b0, b1);
                store16x16(d[2] + 2 * x, r0, r1);
                break;
            case Deep16Out::PlanarFloat:
                storeFloat8(d[0] + 4 * x, g0, max16, scale);
                storeFloat8(d[0] + 4 * x + 32, g1, max16, scale);
                storeFloat8(d[1] + 4 * x, b0, max16, scale);
                storeFloat8(d[1] + 4 * x + 32, b1, max16, scale);
                storeFloat8(d[2] + 4 * x, r0, max16, scale);
                storeFloat8(d[2] + 4 * x + 32, r1, max16, scale);
                break;
            default:
            {
                // packus + 0xD8 leaves pixels 0-7 in the low half, 8-15 high.
                const __m256i R = _mm256_permute4x64_epi64(_mm256_packus_epi32(r0, r1), 0xD8);
                const __m256i G = _mm256_permute4x64_epi64(_mm256_packus_epi32(g0, g1), 0xD8);
                const __m256i B = _mm256_permute4x64_epi64(_mm256_packus_epi32(b0, b1), 0xD8);
                uint8_t* o = d[0] + size_t(outBytes) * x;
                storePacked8(p, o, _mm256_castsi256_si128(R), _mm256_castsi256_si128(G),
                             _mm256_castsi256_si128(B), masks3);
                storePacked8(p, o + 8 * outBytes, _mm256_extracti128_si256(R, 1),
                             _mm256_extracti128_si256(G, 1), _mm256_extracti128_si256(B, 1), masks3);
                break;
            }
            }
        }
    }
    return x;
}

NELUX_AVX2 void convertAvx2(const RgbToRgbPlan& p, const uint8_t* src, ptrdiff_t srcStride,
                            uint8_t* const dst[4], const int dstStride[4], int y0, int y1)
{
    for (int y = y0; y < y1; y++)
    {
        const uint8_t* s = src + srcStride * y;
        uint8_t* d[4];
        rowPointers(p, dst, dstStride, y, d);
        int x = 0;
        switch (p.kind)
        {
        case RgbKind::Copy:
            x = 0; // the scalar model's memcpy is the kernel
            break;
        case RgbKind::Packed:
            x = p.dstBytes == 3 ? packed3Avx2(p, s, d[0], p.width) : packed4Avx2(p, s, d[0], p.width);
            break;
        case RgbKind::Planar8:
            x = planar8Avx2(p, s, d, p.width);
            break;
        case RgbKind::PlanarDeep:
            x = deepAvx2<false>(p, s, d, p.width);
            break;
        case RgbKind::Deep16:
            x = deepAvx2<true>(p, s, d, p.width);
            break;
        }
        rowScalar(p, s, d, x, p.width);
    }
}

void convertScalar(const RgbToRgbPlan& p, const uint8_t* src, ptrdiff_t srcStride,
                   uint8_t* const dst[4], const int dstStride[4], int y0, int y1)
{
    for (int y = y0; y < y1; y++)
    {
        uint8_t* d[4];
        rowPointers(p, dst, dstStride, y, d);
        rowScalar(p, src + srcStride * y, d, 0, p.width);
    }
}

#endif // NELUX_RGB2RGB_X86

} // namespace

std::unique_ptr<RgbToRgbPlan> makeRgbToRgbPlan(AVPixelFormat srcFmt, AVPixelFormat dstFmt,
                                               int width, int height, int swsFlags)
{
#if !NELUX_RGB2RGB_X86
    (void)srcFmt, (void)dstFmt, (void)width, (void)height, (void)swsFlags;
    return nullptr;
#else
    // Unscaled RGB -> RGB never reads the flags' filter, but they pick other
    // paths (SWS_FULL_CHR_H_INP, ACCURATE_RND, BITEXACT, dither) for the deep
    // kind; only the plain bilinear context nelux builds is modelled.
    if (srcFmt != AV_PIX_FMT_RGB24 || width <= 0 || height <= 0 || swsFlags != SWS_BILINEAR)
        return nullptr;
    auto p = std::make_unique<RgbToRgbPlan>();
    p->srcFmt = srcFmt;
    p->dstFmt = dstFmt;
    p->width = width;
    p->height = height;
    switch (dstFmt)
    {
    case AV_PIX_FMT_RGB24:
        p->kind = RgbKind::Copy;
        break;
    case AV_PIX_FMT_GBRP:
    case AV_PIX_FMT_GBRAP:
        p->kind = RgbKind::Planar8;
        p->alphaPlane = dstFmt == AV_PIX_FMT_GBRAP;
        break;
    case AV_PIX_FMT_GBRP9LE:
    case AV_PIX_FMT_GBRP10LE:
    case AV_PIX_FMT_GBRP12LE:
    case AV_PIX_FMT_GBRP14LE:
        p->kind = RgbKind::PlanarDeep;
        p->depth = av_pix_fmt_desc_get(dstFmt)->comp[0].depth;
        p->sh = 22 + 8 - p->depth;
        fillDeepCoefficients(*p);
        break;
    case AV_PIX_FMT_GBRP16LE:
    case AV_PIX_FMT_GBRPF32LE:
    case AV_PIX_FMT_RGB48LE:
    case AV_PIX_FMT_BGR48LE:
    case AV_PIX_FMT_RGBA64LE:
    case AV_PIX_FMT_BGRA64LE:
        p->kind = RgbKind::Deep16;
        p->deepOut = dstFmt == AV_PIX_FMT_GBRP16LE    ? Deep16Out::Planar
                     : dstFmt == AV_PIX_FMT_GBRPF32LE ? Deep16Out::PlanarFloat
                     : dstFmt == AV_PIX_FMT_RGB48LE || dstFmt == AV_PIX_FMT_BGR48LE
                         ? Deep16Out::Packed3
                         : Deep16Out::Packed4;
        p->swapRB = dstFmt == AV_PIX_FMT_BGR48LE || dstFmt == AV_PIX_FMT_BGRA64LE;
        p->depth = 16;
        fillDeepCoefficients(*p);
        break;
    default:
        if (!packedOrder(dstFmt, *p))
            return nullptr;
        p->kind = RgbKind::Packed;
        break;
    }
    p->fn = cpuHasAvx2() ? convertAvx2 : convertScalar;
    return p;
#endif
}

void convertRgbReference(const RgbToRgbPlan& plan, const uint8_t* src, ptrdiff_t srcStride,
                         uint8_t* const dst[4], const int dstStride[4], int y0, int y1)
{
#if NELUX_RGB2RGB_X86
    convertScalar(plan, src, srcStride, dst, dstStride, y0, y1);
#else
    (void)plan, (void)src, (void)srcStride, (void)dst, (void)dstStride, (void)y0, (void)y1;
#endif
}

} // namespace simd
} // namespace cpu
} // namespace conversion
} // namespace nelux
