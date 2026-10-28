#pragma once

extern "C"
{
#include <libavutil/pixfmt.h>
}

#include <cstddef>
#include <cstdint>
#include <memory>

namespace nelux
{
namespace conversion
{
namespace cpu
{
namespace simd
{

/**
 * Byte-exact AVX2 stand-in for the libswscale pass RGBToAutoConverter runs
 * for an RGB DESTINATION at the source size (it sets no colorspace details
 * there): rgb24 into
 *
 *  - another 8-bit RGB layout (rgb24, bgr24, rgba/bgra/argb/abgr, rgb0/bgr0/
 *    0rgb/0bgr, gbrp, gbrap). swscale's unscaled wrappers copy or shuffle
 *    bytes there, so the kernel is a copy or a shuffle; padding and alpha
 *    bytes come out 255 as swscale writes them.
 *  - planar RGB of 9 to 14 bits (gbrp9le .. gbrp14le: r210, r10k, avrp, dpx
 *    10/12-bit, dnxhr_444, ...). swscale has no unscaled path for those and
 *    runs its full scaler, which goes through its internal limited-range
 *    BT.601 YUV and back: the output is NOT a bit expansion of the input
 *    (an 8-bit grey 0 can come out as 0 or 1 depending on the other
 *    channels). The pass is pointwise (identity filters, no dither), so the
 *    kernel reproduces it per pixel: rgb24ToY/rgb24ToUV, the identity
 *    hscale doubling, then yuv2gbrp_full_X with the yuv2rgb coefficients
 *    ff_yuv2rgb_c_init_tables derives for SWS_CS_DEFAULT, limited range.
 *  - 16-bit and float RGB (gbrp16le, rgb48le/bgr48le, rgba64le/bgra64le,
 *    gbrpf32le: dpx 16-bit, exr, hdr, pfm, phm, ...): the same round trip
 *    with swscale's 19-bit intermediates (hscale16to19), ending in
 *    yuv2gbrp16_full_X / yuv2gbrpf32_full_X or, for the packed layouts,
 *    yuv2rgba64_full_1 (one-tap vertical filters, uvalpha 0).
 *
 * Mirrors FFmpeg 9.0.2 (libswscale 10.1.102) on x86-64. The exhaustive suite
 * (all 2^24 colours, every geometry class) proves it and has to be re-run
 * whenever tools/ffmpeg.lock moves; RGBToAutoConverter also checks it against
 * sws_scale once per geometry, and tests/test_simd_rgb_to_rgb.py holds it to
 * the converter's sws_scale output.
 */

struct RgbToRgbPlan;

/// Converts rows [y0, y1) of the frame from @p src (rgb24) into @p dst.
/// Strides are in bytes and may be anything; nothing outside [0, width) x
/// [y0, y1) of the destination planes is written.
using RgbToRgbFn = void (*)(const RgbToRgbPlan& plan, const uint8_t* src,
                            ptrdiff_t srcStride, uint8_t* const dst[4],
                            const int dstStride[4], int y0, int y1);

/// The plan for rgb24 -> @p dstFmt at width x height, or nullptr when the
/// conversion must stay on swscale (other source, resize, unsupported
/// destination). On a CPU without AVX2 the plan's kernel is the scalar model.
std::unique_ptr<RgbToRgbPlan> makeRgbToRgbPlan(AVPixelFormat srcFmt, AVPixelFormat dstFmt,
                                               int width, int height, int swsFlags);

/// Scalar model of the same pass (the AVX2 kernel's reference and row tails).
void convertRgbReference(const RgbToRgbPlan& plan, const uint8_t* src, ptrdiff_t srcStride,
                         uint8_t* const dst[4], const int dstStride[4], int y0, int y1);

enum class RgbKind
{
    Copy,        // rgb24 -> rgb24
    Packed,      // rgb24 -> 3/4-byte packed order (filler 255)
    Planar8,     // rgb24 -> gbrp / gbrap
    PlanarDeep,  // rgb24 -> gbrp9..14 through swscale's YUV round trip
    Deep16,      // rgb24 -> 16-bit / float RGB through the same round trip
};

// Deep16 destination layouts.
enum class Deep16Out
{
    Planar,      // gbrp16le
    PlanarFloat, // gbrpf32le (value / 65535 as float)
    Packed3,     // rgb48le / bgr48le
    Packed4,     // rgba64le / bgra64le (alpha 0xFFFF)
};

struct RgbToRgbPlan
{
    AVPixelFormat srcFmt = AV_PIX_FMT_NONE;
    AVPixelFormat dstFmt = AV_PIX_FMT_NONE;
    int width = 0;
    int height = 0;
    RgbKind kind = RgbKind::Copy;

    // Packed: destination bytes per pixel and, per destination byte, the
    // source channel it takes (0 R, 1 G, 2 B) or -1 for a 255 filler.
    int dstBytes = 3;
    int8_t order[4] = {0, 1, 2, -1};
    bool alphaPlane = false;  // gbrap: a fourth plane of 255

    // PlanarDeep: output depth, yuv2gbrp's shift (22 + 8 - depth) and the
    // coefficients (rgb2yuv: RGB2YUV_SHIFT 15; yuv2rgb: 13-bit, as
    // SwsInternal holds them).
    Deep16Out deepOut = Deep16Out::Planar;
    bool swapRB = false; // bgr48le / bgra64le
    int depth = 8;
    int sh = 22;
    int32_t ry = 0, gy = 0, by = 0, ru = 0, gu = 0, bu = 0, rv = 0, gv = 0, bv = 0;
    int32_t yOffset = 0, yCoeff = 0, v2r = 0, v2g = 0, u2g = 0, u2b = 0;

    RgbToRgbFn fn = nullptr;
};

} // namespace simd
} // namespace cpu
} // namespace conversion
} // namespace nelux
