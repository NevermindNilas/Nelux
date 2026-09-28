#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>
#include <array>

namespace nelux {

// Owns an independent demuxer and a presentation-order index. It never changes
// the streaming reader's position. The index is decoded lazily without RGB work.
class IndexedDecoder {
public:
    struct Index {
        std::string path;
        uintmax_t fileSize = 0;
        int64_t modified = 0;
        std::vector<int64_t> pts, anchors;
        std::vector<std::pair<double, double>> timing;
        int timeBaseNum = 0, timeBaseDen = 1;
        bool canSeek = false, gpuCanSeek = false;
        int streamIndex = -1;
    };
    struct Config {
        std::string path;
        int threads;
        int width;
        int height;
        int channels;
        bool force8Bit;
        int resizeFilter;
        torch::ScalarType dtype;
        torch::Device device;
        int streamIndex = -1;
        bool asyncFrames = false;
    };
    explicit IndexedDecoder(Config config);
    ~IndexedDecoder();
    int64_t size();
    std::vector<std::pair<double, double>> timing();
    std::vector<std::pair<double, double>> timingFor(const std::vector<int64_t>& indices);
    std::vector<int64_t> indicesPlayedAt(const std::vector<double>& seconds);
    std::vector<int64_t> clipIndicesPlayedAt(const std::vector<double>& starts,
                                            int64_t length, double stride,
                                            const std::string& policy);
    std::shared_ptr<Index> getIndex();
    void setIndex(std::shared_ptr<Index> index);
    std::array<int64_t, 5> stats() const;
    torch::Tensor decode(const std::vector<int64_t>& indices);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace nelux
