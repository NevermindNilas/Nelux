#pragma once
#include <Tensor.hpp>
#include <optional>
#include <string>

namespace nelux {
// Decodes the first picture of a still image once, with the streaming reader's
// decoder settings and converter. Without encoded bytes, source is opened as a
// reader opens it; with them, source is the file name FFmpeg would have probed.
// Returns nullopt for anything outside that fast path (non-image demuxers,
// display matrices, any failure); the caller then uses the reader, which also
// owns the error reporting.
std::optional<nelux::Tensor> decodeImageCPU(const std::optional<std::string>& encoded, const std::string& source,
                                            int channels, bool force8Bit);
}
