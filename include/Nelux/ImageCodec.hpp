#pragma once
#include <Tensor.hpp>
#include <array>
#include <optional>
#include <string>
#include <vector>
#include <memory>
namespace nelux {
// Python owns one cache per thread, so CUDA cleanup precedes OS thread teardown.
class ImageCodecCache {
public:
    ImageCodecCache();
    ~ImageCodecCache();
    void clear();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend std::vector<nelux::Tensor> decodeJpegsCUDA(const std::vector<std::string>&,
        const std::vector<std::optional<std::array<int, 4>>>&, int, ImageCodecCache&);
    friend nelux::Tensor encodeJpegCUDA(nelux::Tensor, int, int, ImageCodecCache&);
};
// One optional (left, top, width, height) region per image: only that region is
// decoded and returned. Images and regions are validated before any decoding.
std::vector<nelux::Tensor> decodeJpegsCUDA(const std::vector<std::string>& encoded,
    const std::vector<std::optional<std::array<int, 4>>>& regions, int device, ImageCodecCache& cache);
// subsampling is 444, 422 or 420.
nelux::Tensor encodeJpegCUDA(nelux::Tensor image, int quality, int subsampling, ImageCodecCache& cache);
}
