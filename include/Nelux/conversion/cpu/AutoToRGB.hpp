#pragma once

extern "C"
{
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include "Frame.hpp"
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nelux
{
namespace conversion
{
namespace cpu
{

/**
 * @brief Robust color-accurate converter dynamically handling pixel formats to RGB24.
 */
class AutoToRGBConverter
{
  public:
    AutoToRGBConverter()
        : sws_ctx(nullptr), last_src_fmt(AV_PIX_FMT_NONE),
          last_dst_fmt(AV_PIX_FMT_NONE), last_src_colorspace(AVCOL_SPC_UNSPECIFIED),
          last_src_color_range(AVCOL_RANGE_UNSPECIFIED), last_width(0), last_height(0),
          last_out_width(0), last_out_height(0),
                    force_8bit(false), out_width(0), out_height(0),
          last_resize_flags_(SWS_BILINEAR)
    {
    }

    ~AutoToRGBConverter()
    {
        if (sws_ctx)
            sws_freeContext(sws_ctx);
        if (rgb10_ctx)
            sws_freeContext(rgb10_ctx);
    }

    void setForce8Bit(bool enabled) { force_8bit = enabled; }

    // Select grayscale output (single-channel GRAY8 / GRAY16LE) instead of the
    // default 3-channel RGB24 / RGB48LE. libswscale derives luma from the
    // source colorspace/range exactly as it does for the RGB path, so the
    // grayscale plane is BT.601/709-correct rather than a naive channel average.
    void setGrayscale(bool enabled) { channels_ = enabled ? 1 : 3; }

    // Output channel count: 1 = GRAY8/GRAY16LE, 3 = RGB24/RGB48LE (default),
    // 4 = RGBA/RGBA64LE. The 4-channel form is what makes an alpha-bearing
    // source (ProRes 4444 yuva444p10le/yuva444p12le, VP9/PNG with alpha)
    // reachable at all; libswscale carries the alpha plane straight through
    // (ProRes alpha is straight, not premultiplied) and synthesises an opaque
    // plane when the source has none, exactly as `ffmpeg -pix_fmt rgba` does.
    void setOutputChannels(int channels)
    {
        if (channels != 1 && channels != 3 && channels != 4)
            throw std::runtime_error(
                "AutoToRGBConverter: expected 1 (gray), 3 (RGB) or 4 (RGBA) output "
                "channels, got " + std::to_string(channels));
        channels_ = channels;
    }

    // Set decoder-side output dimensions. Pass (0, 0) to disable (use source dims).
    // When both are > 0, sws_scale will resize the frame to these dimensions and
    // the RGB-passthrough fast path is bypassed.
    void setOutputSize(int width, int height)
    {
        out_width = (width > 0 && height > 0) ? width : 0;
        out_height = (width > 0 && height > 0) ? height : 0;
    }

    // Select the libswscale scaling kernel (SWS_BILINEAR / SWS_BICUBIC /
    // SWS_LANCZOS / ...). This governs spatial rescaling only; it is honored
    // exclusively when an actual resize is active (see convert()). Changing it
    // (like changing the output size) invalidates the cached sws context, so
    // it may be set at any time and takes effect on the next convert().
    void setResizeFilter(int swsFlags)
    {
        if (swsFlags > 0)
            resize_flags_ = swsFlags;
    }

    void convert(nelux::Frame& frame, void* buffer)
    {
        AVFrame* av_frame = frame.get();

        // Input validation
        if (!av_frame || !av_frame->data[0])
        {
            throw std::runtime_error("AutoToRGBConverter: Invalid input frame (null frame or data)");
        }
        if (!buffer)
        {
            throw std::runtime_error("AutoToRGBConverter: Null output buffer");
        }

        const AVPixelFormat src_fmt = frame.getPixelFormat();
        const int width = frame.getWidth();
        const int height = frame.getHeight();

        // Validate dimensions
        if (width <= 0 || height <= 0)
        {
            throw std::runtime_error("AutoToRGBConverter: Invalid frame dimensions");
        }

        // Decoder-side resize target. 0 means: output matches source dims.
        const int dst_w = (out_width  > 0) ? out_width  : width;
        const int dst_h = (out_height > 0) ? out_height : height;
        const bool resize_active = (out_width > 0 && out_height > 0) &&
                                   (out_width != width || out_height != height);

        // 1) Derive effective bit depth from the frame itself
        const int bit_depth = effective_bit_depth_from_frame(av_frame);

        // 1.5) Colorspace passthrough. We intentionally do NOT force a default
        // when the frame is tagged UNSPECIFIED — libswscale's UNSPECIFIED
        // handling (BT.601 coefficients) matches ffmpeg / torchcodec, and any
        // height-based fallback here would silently disagree with them on
        // untagged HD clips (was a ~33 VMAF point gap; see
        // tests/output/pixfmt_matrix/REPORT.md "untagged" row).
        AVColorSpace src_colorspace = av_frame->colorspace;

        AVColorRange src_color_range = av_frame->color_range;
        if (src_color_range == AVCOL_RANGE_UNSPECIFIED)
            src_color_range = AVCOL_RANGE_MPEG;

        // 1.8) Fast-path for RGB24 input. Direct memcpy per row — no swscale
        // overhead, no libyuv dep. Skip when resizing or when grayscale output
        // is requested (RGB24 -> GRAY8 still needs a luma conversion).
        if (!resize_active && channels_ == 3 && bit_depth <= 8 &&
            src_fmt == AV_PIX_FMT_RGB24)
        {
            uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
            int dst_stride = width * 3;
            for (int i = 0; i < height; ++i) {
                std::memcpy(dst_ptr + i*dst_stride,
                            av_frame->data[0] + i*av_frame->linesize[0],
                            width * 3);
            }
            return;
        }

        // 1.9) Fast-path for grayscale output when the source is already the
        // matching single-plane gray format and is full-range. A direct plane
        // copy is exact (no swscale rounding, so gray/gray16 data such as depth
        // maps round-trips losslessly) and much cheaper than the swscale
        // gray->gray path. A limited-range source still needs swscale to expand
        // to full range, so it is excluded here.
        if (channels_ == 1 && !resize_active && src_color_range == AVCOL_RANGE_JPEG)
        {
            const bool srcEightBit = (bit_depth <= 8 || force_8bit);
            const AVPixelFormat wantSrc =
                srcEightBit ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_GRAY16LE;
            if (src_fmt == wantSrc)
            {
                const int rowBytes = width * (srcEightBit ? 1 : 2);
                uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
                for (int i = 0; i < height; ++i)
                    std::memcpy(dst_ptr + static_cast<size_t>(i) * rowBytes,
                                av_frame->data[0] +
                                    static_cast<size_t>(i) * av_frame->linesize[0],
                                rowBytes);
                return;
            }
            if (!srcEightBit && src_fmt == AV_PIX_FMT_GRAY16BE)
            {
                // PNG stores 16-bit gray samples big-endian. swscale's
                // GRAY16BE -> GRAY16LE path changes some sample values even
                // with full-range metadata; a byte swap preserves depth data.
                uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
                for (int i = 0; i < height; ++i)
                {
                    const uint8_t* src_row = av_frame->data[0] +
                                             static_cast<std::ptrdiff_t>(i) * av_frame->linesize[0];
                    uint8_t* dst_row = dst_ptr + static_cast<size_t>(i) * width * 2;
                    for (int x = 0; x < width; ++x)
                    {
                        dst_row[2 * x] = src_row[2 * x + 1];
                        dst_row[2 * x + 1] = src_row[2 * x];
                    }
                }
                return;
            }
        }

        // 1.10) Gray from 8-bit planar YUV, no resize: Y-plane only (skip U/V).
        // Full-range (JPEG, incl. YUVJ*): Y is already full-range, memcpy.
        // Limited-range (MPEG): Y 16..235 -> 0..255 via LUT matching swscale's
        // full-range gray expansion byte-exactly (verified vs ffmpeg rgb24/gray
        // on yuv420p/422p/444p/nv12 across odd widths). Resize and >8-bit fall
        // back to swscale below.
        if (channels_ == 1 && !resize_active && bit_depth <= 8 && std::getenv("NELUX_ENABLE_YUV_GRAY_FAST") && !std::getenv("NELUX_DISABLE_GRAY_FAST"))
        {
            const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(src_fmt);
            const bool isYUV8 = d && d->nb_components == 3 &&
                d->comp[0].depth == 8 && !(d->flags & AV_PIX_FMT_FLAG_RGB) &&
                (src_fmt == AV_PIX_FMT_YUV420P || src_fmt == AV_PIX_FMT_YUVJ420P ||
                 src_fmt == AV_PIX_FMT_YUV422P || src_fmt == AV_PIX_FMT_YUVJ422P ||
                 src_fmt == AV_PIX_FMT_YUV444P || src_fmt == AV_PIX_FMT_YUVJ444P ||
                 src_fmt == AV_PIX_FMT_NV12 || src_fmt == AV_PIX_FMT_NV21 ||
                 src_fmt == AV_PIX_FMT_YUV410P || src_fmt == AV_PIX_FMT_YUV411P ||
                 src_fmt == AV_PIX_FMT_YUV440P);
            if (isYUV8)
            {
                uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
                const uint8_t* y_src = av_frame->data[0];
                const int y_ls = av_frame->linesize[0];
                if (src_color_range == AVCOL_RANGE_JPEG)
                {
                    for (int y = 0; y < height; ++y)
                        std::memcpy(dst_ptr + static_cast<size_t>(y) * width,
                                    y_src + static_cast<size_t>(y) * y_ls, width);
                    return;
                }
                else
                {
                    // Limited-range LUT: (Y-16)*255/219 with round-half-up.
                    // Static per-process (same for BT.601/709 gray: luma-only).
                    static uint8_t lut[256];
                    static bool lutInit = false;
                    if (!lutInit)
                    {
                        for (int i = 0; i < 256; ++i)
                        {
                            int v;
                            if (i <= 16) v = 0;
                            else if (i >= 235) v = 255;
                            else v = ((i - 16) * 255 + 109) / 219;
                            lut[i] = static_cast<uint8_t>(v);
                        }
                        lutInit = true;
                    }
                    for (int y = 0; y < height; ++y)
                    {
                        const uint8_t* s = y_src + static_cast<size_t>(y) * y_ls;
                        uint8_t* dd = dst_ptr + static_cast<size_t>(y) * width;
                        for (int x = 0; x < width; ++x)
                            dd[x] = lut[s[x]];
                    }
                    return;
                }
            }
            // Packed RGB -> luma kernel (BT.601 full-range, matches swscale's
            // default for untagged RGB: Y=(77*R+150*G+29*B+128)>>8).
            // Byte-exact vs swscale SWS_BILINEAR for RGB24/BGR24; other packed
            // formats fall back to swscale.
            if (src_fmt == AV_PIX_FMT_RGB24 || src_fmt == AV_PIX_FMT_BGR24)
            {
                const bool isBGR = (src_fmt == AV_PIX_FMT_BGR24);
                uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
                const uint8_t* s0 = av_frame->data[0];
                const int sls = av_frame->linesize[0];
                for (int y = 0; y < height; ++y)
                {
                    const uint8_t* s = s0 + static_cast<size_t>(y) * sls;
                    uint8_t* dd = dst_ptr + static_cast<size_t>(y) * width;
                    for (int x = 0; x < width; ++x)
                    {
                        const uint8_t r = isBGR ? s[3*x+2] : s[3*x];
                        const uint8_t g = s[3*x+1];
                        const uint8_t b = isBGR ? s[3*x] : s[3*x+2];
                        dd[x] = static_cast<uint8_t>((77*r + 150*g + 29*b + 128) >> 8);
                    }
                }
                return;
            }
        }

        // 1.11) Opaque RGBA from packed RGB 8-bit, no resize, source has no
        // alpha: RGB24 + 0xFF fill. Gated on !has_alpha_from_frame() so
        // alpha-bearing sources (ProRes 4444, VP9/PNG with alpha) keep the
        // swscale path that carries the plane through. YUV->RGBA opaque
        // stays on swscale (byte-exact, same opaque synthesis as ffmpeg).
        if (channels_ == 4 && !resize_active && (bit_depth <= 8 || force_8bit))
        {
            const bool srcHasAlpha = has_alpha_from_frame(av_frame);
            if (!srcHasAlpha &&
                (src_fmt == AV_PIX_FMT_RGB24 || src_fmt == AV_PIX_FMT_BGR24))
            {
                const bool isBGR = (src_fmt == AV_PIX_FMT_BGR24);
                uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
                const uint8_t* s0 = av_frame->data[0];
                const int sls = av_frame->linesize[0];
                for (int y = 0; y < height; ++y)
                {
                    const uint8_t* s = s0 + static_cast<size_t>(y) * sls;
                    uint8_t* dd = dst_ptr + static_cast<size_t>(y) * width * 4;
                    for (int x = 0; x < width; ++x)
                    {
                        const uint8_t r = isBGR ? s[3*x+2] : s[3*x];
                        const uint8_t g = s[3*x+1];
                        const uint8_t b = isBGR ? s[3*x] : s[3*x+2];
                        dd[4*x] = r; dd[4*x+1] = g; dd[4*x+2] = b; dd[4*x+3] = 0xFF;
                    }
                }
                return;
            }
        }

        // 1.12) 10-bit fast paths, gated on bit_depth==10 && eightBit && !resize.
        // BE sources reuse the byte-swap below (same as the gray16BE path).
        {
            const bool eightBitLocal = (bit_depth <= 8 || force_8bit);
            const bool is10 = (bit_depth == 10);
            if (is10 && eightBitLocal && !resize_active)
            {
                auto isBE = [&]() -> bool {
                    const char* n = av_get_pix_fmt_name(src_fmt);
                    if (!n) return false;
                    std::string s(n);
                    return s.size() >= 2 && s.compare(s.size()-2, 2, "be") == 0;
                };
                // Gray10 -> Gray8 over Y only: (y+32)>>6 would be for
                // left-aligned storage; yuv420p10le-class planar stores
                // 10-bit in the low bits (0..1023), so full-range is
                // (y+2)>>2 and limited-range is a 64..940 -> 0..255 LUT.
                // Both skip U/V entirely. BE swaps first.
                if (channels_ == 1)
                {
                    const AVPixFmtDescriptor* d10 = av_pix_fmt_desc_get(src_fmt);
                    const bool isPlanar10 = d10 && d10->comp[0].depth == 10 &&
                        (src_fmt == AV_PIX_FMT_YUV420P10LE || src_fmt == AV_PIX_FMT_YUV420P10BE ||
                         src_fmt == AV_PIX_FMT_YUV422P10LE || src_fmt == AV_PIX_FMT_YUV422P10BE ||
                         src_fmt == AV_PIX_FMT_YUV444P10LE || src_fmt == AV_PIX_FMT_YUV444P10BE ||
                         src_fmt == AV_PIX_FMT_NV12 || src_fmt == AV_PIX_FMT_YUV420P10LE);
                    // Note: NV12 is 8-bit; listed for shape symmetry but bit_depth
                    // gate above already excludes it. Kept explicit, not taken.
                    const bool wantY10 = (src_fmt == AV_PIX_FMT_YUV420P10LE ||
                        src_fmt == AV_PIX_FMT_YUV420P10BE ||
                        src_fmt == AV_PIX_FMT_YUV422P10LE || src_fmt == AV_PIX_FMT_YUV422P10BE ||
                        src_fmt == AV_PIX_FMT_YUV444P10LE || src_fmt == AV_PIX_FMT_YUV444P10BE);
                    if (wantY10)
                    {
                        const bool be = isBE();
                        uint8_t* dst_ptr = static_cast<uint8_t*>(buffer);
                        const uint8_t* yb = av_frame->data[0];
                        const int yls = av_frame->linesize[0];
                        (void)isPlanar10;
                        if (src_color_range == AVCOL_RANGE_JPEG)
                        {
                            for (int y = 0; y < height; ++y)
                            {
                                const uint8_t* row = yb + static_cast<size_t>(y) * yls;
                                uint8_t* dd = dst_ptr + static_cast<size_t>(y) * width;
                                for (int x = 0; x < width; ++x)
                                {
                                    uint16_t v;
                                    if (be) v = (static_cast<uint16_t>(row[2*x]) << 8) | row[2*x+1];
                                    else v = static_cast<uint16_t>(row[2*x]) | (static_cast<uint16_t>(row[2*x+1]) << 8);
                                    v &= 0x3FF;
                                    dd[x] = static_cast<uint8_t>((v + 2) >> 2);
                                }
                            }
                            return;
                        }
                        else
                        {
                            for (int y = 0; y < height; ++y)
                            {
                                const uint8_t* row = yb + static_cast<size_t>(y) * yls;
                                uint8_t* dd = dst_ptr + static_cast<size_t>(y) * width;
                                for (int x = 0; x < width; ++x)
                                {
                                    uint16_t v;
                                    if (be) v = (static_cast<uint16_t>(row[2*x]) << 8) | row[2*x+1];
                                    else v = static_cast<uint16_t>(row[2*x]) | (static_cast<uint16_t>(row[2*x+1]) << 8);
                                    v &= 0x3FF;
                                    int o;
                                    if (v <= 64) o = 0;
                                    else if (v >= 940) o = 255;
                                    else o = ((static_cast<int>(v) - 64) * 255 + 438) / 876;
                                    dd[x] = static_cast<uint8_t>(o);
                                }
                            }
                            return;
                        }
                    }
                }
                // RGB10 -> RGB8 via RGB48LE then pack (AVX2-friendly).
                // swscale's direct YUV10->RGB24 truncates (47dB vs ffmpeg);
                // routing through RGB48LE preserves precision and the final
                // (v+128)>>8 rounding reaches >=60dB deterministically.
                // Gated on 3-channel 8-bit no-resize (gray handled above,
                // RGBA opaque stays on swscale for byte-exact alpha).
                if (channels_ == 3 && std::getenv("NELUX_ENABLE_RGB10"))
                {
                    uint8_t* dst8 = static_cast<uint8_t*>(buffer);
                    const int64_t px = static_cast<int64_t>(width) * height;
                    // Temp RGB48LE (3x uint16 per pixel). Thread-local reused
                    // to avoid per-frame alloc; grows to the largest frame.
                    thread_local std::vector<uint16_t> tmp48;
                    const size_t need48 = static_cast<size_t>(px) * 3;
                    if (tmp48.size() < need48)
                        tmp48.resize(need48);
                    // (Re)build the RGB48 context on geometry/colorspace change.
                    if (!rgb10_ctx || rgb10_src_fmt != src_fmt ||
                        rgb10_src_cs != src_colorspace ||
                        rgb10_src_range != src_color_range ||
                        rgb10_w != width || rgb10_h != height)
                    {
                        rgb10_ctx = sws_getCachedContext(rgb10_ctx,
                            width, height, src_fmt,
                            width, height, AV_PIX_FMT_RGB48LE,
                            SWS_BILINEAR, nullptr, nullptr, nullptr);
                        if (!rgb10_ctx)
                            throw std::runtime_error("Failed to init rgb10 swsContext");
                        AVColorSpace cc = src_colorspace;
                        if (cc == AVCOL_SPC_UNSPECIFIED)
                            cc = AVCOL_SPC_BT470BG;
                        const int* sc = sws_getCoefficients(cc);
                        const int* dc = sws_getCoefficients(AVCOL_SPC_BT709);
                        const int sr = (src_color_range == AVCOL_RANGE_JPEG) ? 1 : 0;
                        if (sws_setColorspaceDetails(rgb10_ctx, sc, sr, dc,
                                                     1, 0, 1 << 16, 1 << 16) < 0)
                            throw std::runtime_error("rgb10 colorspace setup failed");
                        rgb10_src_fmt = src_fmt;
                        rgb10_src_cs = src_colorspace;
                        rgb10_src_range = src_color_range;
                        rgb10_w = width; rgb10_h = height;
                    }
                    const uint8_t* sD[4] = {av_frame->data[0], av_frame->data[1],
                                            av_frame->data[2], av_frame->data[3]};
                    const int sL[4] = {av_frame->linesize[0], av_frame->linesize[1],
                                       av_frame->linesize[2], av_frame->linesize[3]};
                    uint8_t* dD[4] = {reinterpret_cast<uint8_t*>(tmp48.data()),
                                      nullptr, nullptr, nullptr};
                    const int dL[4] = {width * 3 * 2, 0, 0, 0};
                    const int got = sws_scale(rgb10_ctx, sD, sL, 0, height, dD, dL);
                    if (got != height)
                        throw std::runtime_error("rgb10 sws_scale failed");
                    // Pack RGB48LE -> RGB24 with rounding. Scalar loop here;
                    // MSVC auto-vectorizes to AVX2 (verified in disasm for
                    // /O2: vpmul/vpadd/vpsrl pattern). Byte order LE assumed
                    // (x86_64); BE swap would be needed on big-endian, which
                    // Windows/x64 never is — swscale already emitted LE.
                    const uint16_t* s48 = tmp48.data();
                    for (int64_t i = 0, n = px * 3; i < n; ++i)
                    {
                        const int rounded = (static_cast<int>(s48[i]) + 128) >> 8;
                        dst8[i] = static_cast<uint8_t>(rounded > 255 ? 255 : rounded);
                    }
                    return;
                }
            }
        }

        // 2) Choose destination format/stride accordingly
        // If force_8bit is true, we want 8-bit output regardless of input bit
        // depth. Otherwise, preserve >8-bit sources with a 16-bit output.
        // channels_ picks the GRAY / RGB / RGBA variant of each.
        const bool eightBit = (bit_depth <= 8 || force_8bit);
        const AVPixelFormat dst_fmt =
            (channels_ == 1) ? (eightBit ? AV_PIX_FMT_GRAY8 : AV_PIX_FMT_GRAY16LE)
          : (channels_ == 4) ? (eightBit ? AV_PIX_FMT_RGBA : AV_PIX_FMT_RGBA64LE)
                             : (eightBit ? AV_PIX_FMT_RGB24 : AV_PIX_FMT_RGB48LE);
        const int elem_size = eightBit ? 1 : 2; // bytes per channel
        const int channels = channels_;


        // 3) (Re)build sws context if anything changed (source/destination
        // geometry, formats, colorspace/range, or the selected scaling kernel —
        // so setResizeFilter()/setOutputSize() after the first convert()
        // invalidate the cache via resize_flags_/dst dims below).
        if (!sws_ctx || src_fmt != last_src_fmt || dst_fmt != last_dst_fmt ||
            src_colorspace != last_src_colorspace ||
            src_color_range != last_src_color_range || width != last_width ||
            height != last_height || dst_w != last_out_width || dst_h != last_out_height ||
            resize_flags_ != last_resize_flags_)
        {
            // Scaling kernel. When resizing, honor the caller-selected filter
            // (resize_flags_, default SWS_BILINEAR). When NOT resizing, force
            // SWS_BILINEAR: the flag then only affects chroma upsampling for
            // subsampled YUV, and plain BILINEAR matches ffmpeg's default chroma
            // path / torchcodec byte-output (SWS_ACCURATE_RND / FULL_CHR would
            // diverge and add latency — see tests/output/pixfmt_matrix/REPORT.md).
            int flags = resize_active ? resize_flags_ : SWS_BILINEAR;

            // Only use POINT (nearest neighbor) when we are just shuffling
            // channels (RGB/BGR/BGRA etc. at identical dims). Using POINT
            // for YUV420P would cause nearest-neighbor chroma upsampling
            // (blocky colors), which we want to avoid. This is a pure function
            // of the CURRENT frame — the old form also keyed on
            // width == last_width, which made the very first convert()
            // (last_width == 0) pick a different kernel than every later one
            // without rebuilding, so the kernel depended on call history.
            if (!resize_active &&
                (src_fmt == AV_PIX_FMT_RGB24 || src_fmt == AV_PIX_FMT_BGR24 ||
                 src_fmt == AV_PIX_FMT_ARGB || src_fmt == AV_PIX_FMT_RGBA ||
                 src_fmt == AV_PIX_FMT_ABGR || src_fmt == AV_PIX_FMT_BGRA))
            {
                flags = SWS_POINT;
            }

            sws_ctx = sws_getCachedContext(
                sws_ctx,
                width,
                height,
                src_fmt,
                dst_w,
                dst_h,
                dst_fmt,
                flags,
                nullptr,
                nullptr,
                nullptr);
            if (!sws_ctx)
                throw std::runtime_error("Failed to initialize swsContext");

            // For sws_setColorspaceDetails we still need a concrete matrix
            // when UNSPECIFIED; pass BT.470BG (BT.601 PAL) which is
            // libswscale's own default.
            AVColorSpace coeff_cs = src_colorspace;
            if (coeff_cs == AVCOL_SPC_UNSPECIFIED)
                coeff_cs = AVCOL_SPC_BT470BG;

            const int* srcCoeffs = sws_getCoefficients(coeff_cs);
            const int* dstCoeffs = sws_getCoefficients(AVCOL_SPC_BT709);
            const int srcRange = (src_color_range == AVCOL_RANGE_JPEG) ? 1 : 0;

            int ok = sws_setColorspaceDetails(sws_ctx, srcCoeffs, srcRange, dstCoeffs,
                                              1, 0, 1 << 16, 1 << 16);
            if (ok < 0)
            {
                throw std::runtime_error(
                    "AutoToRGBConverter: Failed to configure color space details (error=" +
                    std::to_string(ok) + ", colorspace=" + std::to_string(src_colorspace) +
                    ", range=" + std::to_string(src_color_range) + ")");
            }

            last_src_fmt = src_fmt;
            last_dst_fmt = dst_fmt;
            last_src_colorspace = src_colorspace;
            last_src_color_range = src_color_range;
            last_width = width;
            last_height = height;
            last_out_width = dst_w;
            last_out_height = dst_h;
            last_resize_flags_ = resize_flags_;
        }

        // 4) Do the conversion
        const uint8_t* srcData[4] = {av_frame->data[0], av_frame->data[1],
                                     av_frame->data[2], av_frame->data[3]};
        const int srcLineSize[4] = {av_frame->linesize[0], av_frame->linesize[1],
                                    av_frame->linesize[2], av_frame->linesize[3]};

        uint8_t* dstData[4] = {static_cast<uint8_t*>(buffer), nullptr, nullptr,
                               nullptr};
        const int dstLineSize[4] = {dst_w * channels * elem_size, 0, 0, 0};

        const int result =
            sws_scale(sws_ctx, srcData, srcLineSize, 0, height, dstData, dstLineSize);
        if (result != dst_h)
            throw std::runtime_error("sws_scale failed or incomplete");
    }

  private:
    SwsContext* sws_ctx;
    // Separate cached context for the 10-bit two-step (YUV10 -> RGB48LE).
    // Kept apart from sws_ctx so 8-bit and 10-bit files do not thrash one
    // cache entry (dst differs: RGB24 vs RGB48LE).
    SwsContext* rgb10_ctx = nullptr;
    AVPixelFormat rgb10_src_fmt = AV_PIX_FMT_NONE;
    AVColorSpace rgb10_src_cs = AVCOL_SPC_UNSPECIFIED;
    AVColorRange rgb10_src_range = AVCOL_RANGE_UNSPECIFIED;
    int rgb10_w = 0, rgb10_h = 0;
    AVPixelFormat last_src_fmt;
    AVPixelFormat last_dst_fmt;
    AVColorSpace last_src_colorspace;
    AVColorRange last_src_color_range;
    int last_width, last_height;
    int last_out_width, last_out_height;
    bool force_8bit;
    int out_width, out_height;
    // 1 = gray, 3 = RGB (default), 4 = RGBA.
    int channels_ = 3;
    // libswscale scaling kernel used when a resize is active. Default matches
    // the historical hardcoded behavior. Part of the sws cache key
    // (last_resize_flags_), so changing it rebuilds the context.
    int resize_flags_ = SWS_BILINEAR;
    int last_resize_flags_ = SWS_BILINEAR;
};

} // namespace cpu
} // namespace conversion
} // namespace nelux
