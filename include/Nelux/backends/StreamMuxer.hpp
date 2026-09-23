#pragma once

#include <string>

namespace nelux
{

/**
 * Copy the primary video stream from videoSource and the primary audio stream
 * from audioSource into output. Each input timeline is rebased by its stream
 * start_time, as with ffmpeg's default (non-copyts) two-input stream copy.
 * The destination container must support both codecs without transcoding.
 * The completed file replaces output only after the trailer is written.
 */
void mergeStreams(const std::string& videoSource, const std::string& audioSource,
                  const std::string& output);

} // namespace nelux
