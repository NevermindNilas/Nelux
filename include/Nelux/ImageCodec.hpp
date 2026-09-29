#pragma once
#include <torch/torch.h>
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
    friend std::vector<torch::Tensor> decodeJpegsCUDA(const std::vector<std::string>&, int, ImageCodecCache&);
    friend torch::Tensor encodeJpegCUDA(torch::Tensor, int, ImageCodecCache&);
};
std::vector<torch::Tensor> decodeJpegsCUDA(const std::vector<std::string>& encoded, int device, ImageCodecCache& cache);
torch::Tensor encodeJpegCUDA(torch::Tensor image, int quality, ImageCodecCache& cache);
}
