#pragma once
#include <torch/torch.h>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

namespace nelux {
class AudioDecoder {
public:
    AudioDecoder(const std::string& path, int streamIndex, int sampleRate,
                 int channels, int threads);
    ~AudioDecoder();
    std::tuple<torch::Tensor, double, int> samples(double start,
                                                 std::optional<double> end);
    std::tuple<int, int, int64_t, double, std::string, int> metadata();
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
