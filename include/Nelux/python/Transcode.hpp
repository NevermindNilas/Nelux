#pragma once
#ifndef TRANSCODE_HPP
#define TRANSCODE_HPP

#include <optional>
#include <pybind11/pybind11.h>
#include <string>

namespace nelux
{

/**
 * @brief VideoEncoder arguments that re-encode @p source's video the way the
 * ffmpeg CLI would (nelux.transcode).
 *
 * Returns a dict with codec, width, height, fps, pixel_format and options.
 * The pixel format is the source's own when @p codec accepts it, else the
 * closest one it does (avcodec_find_best_pix_fmt_of_list); @p pixelFormat
 * overrides that choice. options carries the source's colour description
 * (colorspace, color_range, color_primaries, color_trc), its chroma sample
 * location while the pixel format is unchanged, and its sample aspect ratio,
 * exactly as `ffmpeg -i src -c:v codec` copies them from the decoded frames.
 * Raises for a source without video and for hardware encoders other than
 * NVENC, which the native path does not drive.
 */
pybind11::dict transcodeSettings(const std::string& source, const std::string& outputPath,
                                 const std::optional<std::string>& codec,
                                 const std::optional<std::string>& pixelFormat);

} // namespace nelux

#endif // TRANSCODE_HPP
