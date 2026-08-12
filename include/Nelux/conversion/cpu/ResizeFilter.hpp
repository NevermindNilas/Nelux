#pragma once

extern "C"
{
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace nelux
{
namespace conversion
{
namespace cpu
{

// Map an ffmpeg swscale scaler name to its SWS_* flag. The accepted names
// mirror ffmpeg's own -sws_flags scaler options so callers can reuse the exact
// vocabulary they already know. Throws std::invalid_argument on an unknown
// name. Shared by the decoder-side resize (VideoReader) and the encoder-side
// resize (VideoEncoder) so both accept the same vocabulary.
inline int swsFlagFromResizeFilter(const std::string& name)
{
    std::string n = name;
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (n.empty() || n == "bilinear")                      return SWS_BILINEAR;
    if (n == "fast_bilinear" || n == "fastbilinear")       return SWS_FAST_BILINEAR;
    if (n == "bicubic")                                    return SWS_BICUBIC;
    if (n == "experimental" || n == "x")                   return SWS_X;
    if (n == "neighbor" || n == "point" || n == "nearest") return SWS_POINT;
    if (n == "area")                                       return SWS_AREA;
    if (n == "bicublin")                                   return SWS_BICUBLIN;
    if (n == "gauss" || n == "gaussian")                   return SWS_GAUSS;
    if (n == "sinc")                                       return SWS_SINC;
    if (n == "lanczos")                                    return SWS_LANCZOS;
    if (n == "spline")                                     return SWS_SPLINE;
    throw std::invalid_argument(
        "Unknown resize_filter: '" + name +
        "'. Valid options: fast_bilinear, bilinear, bicubic, experimental, "
        "neighbor, area, bicublin, gauss, sinc, lanczos, spline.");
}

} // namespace cpu
} // namespace conversion
} // namespace nelux
