#pragma once

extern "C"
{
#include <libavutil/pixfmt.h>
}

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace nelux
{
namespace conversion
{
namespace cpu
{
namespace simd
{

/**
 * Byte-exact AVX2 stand-in for the one libswscale pass RGBToAutoConverter runs
 * when the encoder does not resize: packed RGB (rgb24, rgba, rgb48le,
 * rgba64le) into planar or semi-planar YUV, with exactly the context it builds
 * (sws_getContext(SWS_BILINEAR) followed by sws_setColorspaceDetails with an
 * RGB source, so source range is forced limited and the matrix only touches
 * the destination side).
 *
 * The kernels do not approximate swscale: they reproduce its integer pipeline
 * stage by stage as FFmpeg 9.0.2 runs it on an AVX2 x86-64 CPU — the
 * rgb2yuv table (including the hard-coded BT.601 special case), the 14/15-bit
 * intermediates, the identity hscale doubling, the RangeToJpeg pass for full
 * range, initFilter's bilinear vertical chroma filter, the pmulhw yuv2yuvX
 * approximation on every 8-bit 4:2:0 chroma row but the last (which swscale
 * switches to the C planeX), and the exact 32-bit planeX/nv12cX/p010 paths.
 * Because they mirror swscale internals, the exactness test (test_exact.cpp)
 * has to be re-run whenever tools/ffmpeg.lock moves.
 *
 * Anything outside the supported set returns nullptr from canHandle() and
 * must stay on sws_scale; so does any CPU without AVX2 + OS YMM state.
 */

struct RgbToYuvPlan;

/// Converts luma rows [y0, y1) of the frame (and the chroma rows that belong
/// to them) from @p src into @p dst. Any partition of [0, height) into ranges
/// produces the same bytes as one call over the whole frame, so a caller may
/// split a frame across threads. Strides are in bytes and may be anything
/// (including negative); nothing outside [0, width) x [y0, y1) of the planes
/// is read or written, except the one luma sample past each row that
/// swscale writes too (RgbToYuvPlan::lumaEdgeCopy), when the stride has room.
using RgbToYuvFn = void (*)(const RgbToYuvPlan& plan, const uint8_t* src,
                            ptrdiff_t srcStride, uint8_t* const dst[4],
                            const int dstStride[4], int y0, int y1);

/// True when the CPU has AVX2 and the OS saves YMM state (CPUID + XGETBV).
bool cpuHasAvx2();

/// True when the libswscale loaded at runtime is the one the kernels were
/// proven byte-exact against (10.1.102, FFmpeg 9.0.2). Any other version
/// stays on sws_scale until test_exact has been re-run against it and this
/// version bumped.
bool swscaleIsVerifiedVersion();

/**
 * Function table lookup. Arguments mirror what RGBToAutoConverter derives
 * before it calls libswscale: @p swsColorspace is the SWS_CS_* id it passes to
 * sws_getCoefficients, @p dstRange the 0/1 range it passes to
 * sws_setColorspaceDetails, @p swsFlags the sws_getContext flags. Width and
 * height are the frame size (source and destination are the same size).
 * Returns the AVX2 kernel, or nullptr when the conversion must stay on
 * swscale.
 */
RgbToYuvFn canHandle(AVPixelFormat srcFmt, AVPixelFormat dstFmt, int swsColorspace,
                     int dstRange, int width, int height, int swsFlags);

/// Builds everything a kernel needs for one geometry (coefficients, range
/// constants, chroma filters). Returns nullptr exactly when the conversion is
/// not in the supported set; a CPU without AVX2 still gets a plan (its
/// kernel is then the scalar model, for tests).
std::unique_ptr<RgbToYuvPlan> makePlan(AVPixelFormat srcFmt, AVPixelFormat dstFmt,
                                       int swsColorspace, int dstRange, int width,
                                       int height, int swsFlags);

/// Source rows [r0, r1) a kernel call over luma rows [y0, y1) reads: those
/// rows and, for a vertically subsampled destination, the rows its chroma
/// filter taps around them. Always inside [0, height).
void sourceRowRange(const RgbToYuvPlan& plan, int y0, int y1, int& r0, int& r1);

/// Scalar model of the same swscale pipeline: the reference the AVX2 kernel
/// is tested against, and what fills row tails narrower than a vector.
void convertReference(const RgbToYuvPlan& plan, const uint8_t* src, ptrdiff_t srcStride,
                      uint8_t* const dst[4], const int dstStride[4], int y0, int y1);

struct RgbToYuvPlan
{
    AVPixelFormat srcFmt = AV_PIX_FMT_NONE;
    AVPixelFormat dstFmt = AV_PIX_FMT_NONE;
    int width = 0;
    int height = 0;

    // Source layout: bytes per pixel (3, 4, 6, 8) and whether samples are
    // 16-bit little endian.
    int srcPixelBytes = 3;
    bool deepSource = false;

    // Destination layout.
    int outBits = 8;          // 8, 10 or 12
    int outShift = 0;         // 6 for p010 (MSB-aligned samples)
    bool semiPlanar = false;  // nv12 / p010: interleaved UV plane
    int chromaShiftX = 0;     // log2 horizontal subsampling
    int chromaShiftY = 0;     // log2 vertical subsampling
    int chromaWidth = 0;
    int chromaHeight = 0;

    // Chroma input: true reads pixel pairs (swscale's *ToUV_half, even
    // widths); false reads every pixel and, when horizontally subsampled,
    // runs the hChr bilinear filter below.
    bool halfChroma = false;

    // Odd width into horizontally subsampled 8- to 10-bit planes: swscale's
    // SIMD luma output also writes the sample past the right edge (a copy of
    // the last one) on every row but the last two, and encoders that code
    // whole chroma blocks (ljpeg, yuv4) read it. The kernels write it as well.
    bool lumaEdgeCopy = false;

    // swscale's input_rgb2yuv_table (RGB2YUV_SHIFT = 15).
    int32_t ry = 0, gy = 0, by = 0;
    int32_t ru = 0, gu = 0, bu = 0;
    int32_t rv = 0, gv = 0, bv = 0;

    // RangeToJpeg constants (full-range destination only), as the SIMD
    // range converter applies them: the low 16 bits of the coefficient as a
    // signed pmaddwd factor and the low 32 bits of the offset.
    bool fullRange = false;
    int32_t lumRangeCoeff = 0, lumRangeOffset = 0;
    int32_t chrRangeCoeff = 0, chrRangeOffset = 0;

    // Horizontal chroma filter (odd width into a horizontally subsampled
    // destination); 14-bit coefficients.
    int hFilterSize = 0;
    bool simdHScale16 = true; // deep sources: SSE2 hscale16to15 (biased) vs C
    std::vector<int16_t> hFilter;
    std::vector<int32_t> hFilterPos;

    // Vertical chroma filter (vertically subsampled destination); 12-bit
    // coefficients. vFilterSize == 1 means swscale writes chroma with the
    // plain one-line output functions.
    int vFilterSize = 0;
    std::vector<int16_t> vFilter;
    std::vector<int32_t> vFilterPos;
    // 8-bit planar 4:2:0 only: rows written by yuv2yuvX (pmulhw) rather than
    // the exact C planeX — every chroma row whose luma row is < height - 2.
    bool approxVerticalFilter = false;

    RgbToYuvFn fn = nullptr;
};

} // namespace simd
} // namespace cpu
} // namespace conversion
} // namespace nelux
