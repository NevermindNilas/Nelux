#include <JpegCodec.hpp>
#include <ATen/Parallel.h>
#include <turbojpeg.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace nelux {
namespace {

// One compressor per thread: tj3Init allocates the whole libjpeg state, which
// costs more than encoding a thumbnail.
tjhandle threadCompressor()
{
    struct Handle {
        tjhandle handle = tj3Init(TJINIT_COMPRESS);
        ~Handle() { if (handle) tj3Destroy(handle); }
    };
    thread_local Handle compressor;
    if (!compressor.handle)
        throw std::runtime_error("libjpeg-turbo: failed to create a compressor");
    return compressor.handle;
}

// Runs body(i) for i in [0, n) on up to `width` threads, the caller included.
// The workers are leaked on purpose: joining threads from a static destructor
// deadlocks under the Windows loader lock at process exit.
class StripPool {
public:
    static StripPool& instance()
    {
        static StripPool* pool = new StripPool();
        return *pool;
    }

    void run(int n, int width, const std::function<void(int)>& body)
    {
        width = std::min({width, n, workers_ + 1});
        if (width <= 1) {
            for (int i = 0; i < n; ++i)
                body(i);
            return;
        }
        auto job = std::make_shared<Job>(&body, n);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (int i = 1; i < width; ++i)
                queue_.push_back(job);
        }
        if (width == 2)
            ready_.notify_one();
        else
            ready_.notify_all();
        drain(*job);
        std::unique_lock<std::mutex> lock(job->mutex);
        job->done.wait(lock, [&] { return job->left.load() == 0; });
        if (job->error)
            std::rethrow_exception(job->error);
    }

private:
    struct Job {
        Job(const std::function<void(int)>* body, int n) : body(body), n(n), left(n) {}
        const std::function<void(int)>* body;
        const int n;
        std::atomic<int> next{0};
        std::atomic<int> left;
        std::mutex mutex;
        std::condition_variable done;
        std::exception_ptr error;
    };

    StripPool()
    {
        const unsigned cores = std::thread::hardware_concurrency();
        workers_ = static_cast<int>(std::clamp(cores, 2u, 64u)) - 1;
        for (int i = 0; i < workers_; ++i)
            std::thread([this] { work(); }).detach();
    }

    // Claims indices until none are left. A helper that dequeues a finished
    // job claims nothing and never touches body, which the caller owns.
    static void drain(Job& job)
    {
        for (int i; (i = job.next.fetch_add(1)) < job.n;) {
            try {
                (*job.body)(i);
            } catch (...) {
                std::lock_guard<std::mutex> lock(job.mutex);
                if (!job.error)
                    job.error = std::current_exception();
            }
            if (job.left.fetch_sub(1) == 1) {
                std::lock_guard<std::mutex> lock(job.mutex);
                job.done.notify_all();
            }
        }
    }

    void work()
    {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return !queue_.empty(); });
                job = std::move(queue_.front());
                queue_.pop_front();
            }
            drain(*job);
        }
    }

    int workers_ = 0;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<std::shared_ptr<Job>> queue_;
};

// Worst-case output space for every strip, kept per calling thread so steady
// encoding does not page-fault fresh memory. An outsized one is let go after.
struct Arena {
    std::unique_ptr<unsigned char[]> data;
    size_t capacity = 0;

    unsigned char* reserve(size_t bytes)
    {
        if (bytes > capacity) {
            data.reset();
            data = std::make_unique_for_overwrite<unsigned char[]>(bytes);
            capacity = bytes;
        }
        return data.get();
    }
    void trim()
    {
        constexpr size_t kKeep = size_t(256) << 20;
        if (capacity > kKeep) {
            data.reset();
            capacity = 0;
        }
    }
};

// ~64K pixels per strip: measured fastest from 512x512 to 4K at 16 threads
// (256K left 512x512 serial), and the extra restart markers cost ~0.1% size.
constexpr int64_t kStripPixels = int64_t(1) << 16;

struct StripLayout {
    size_t sofHeight = 0;  // offset of the frame header's 16-bit height
    size_t sos = 0;        // offset of the SOS marker
    size_t data = 0;       // first byte of entropy-coded data
};

StripLayout parseStrip(const unsigned char* jpeg, size_t size)
{
    StripLayout layout;
    size_t pos = 2;  // past SOI
    while (pos + 4 <= size) {
        if (jpeg[pos] != 0xFF)
            break;
        const unsigned marker = jpeg[pos + 1];
        const size_t length = (size_t(jpeg[pos + 2]) << 8) | jpeg[pos + 3];
        if (marker >= 0xC0 && marker <= 0xC3)
            layout.sofHeight = pos + 5;  // FF Cn Lh Ll P Yh Yl
        if (marker == 0xDA) {
            layout.sos = pos;
            layout.data = pos + 2 + length;
            break;
        }
        pos += 2 + length;
    }
    if (!layout.sofHeight || !layout.data || layout.data + 2 > size ||
        jpeg[size - 2] != 0xFF || jpeg[size - 1] != 0xD9)
        throw std::runtime_error("libjpeg-turbo: unexpected strip layout");
    return layout;
}

} // namespace

torch::Tensor encodeJpegCPU(torch::Tensor image, int quality, int subsampling)
{
    if (image.device().type() != torch::kCPU || image.scalar_type() != torch::kUInt8)
        throw std::invalid_argument("JPEG encoding requires a CPU uint8 tensor");
    if (image.dim() != 2 && image.dim() != 3)
        throw std::invalid_argument("image must have a HW or HWC layout");
    const int channels = image.dim() == 2 ? 1 : static_cast<int>(image.size(2));
    if (channels != 1 && channels != 3)
        throw std::invalid_argument("JPEG encoding takes 1 or 3 channels");
    // libjpeg's JPEG_MAX_DIMENSION, below the format's 65535.
    if (image.size(0) <= 0 || image.size(1) <= 0 || image.size(0) > 65500 || image.size(1) > 65500)
        throw std::invalid_argument("JPEG dimensions must be in [1, 65500]");
    if (quality < 1 || quality > 100)
        throw std::invalid_argument("quality must be in [1, 100]");

    int samp = TJSAMP_GRAY;
    if (channels == 3) {
        switch (subsampling) {
        case 444: samp = TJSAMP_444; break;
        case 422: samp = TJSAMP_422; break;
        case 420: samp = TJSAMP_420; break;
        default: throw std::invalid_argument("subsampling must be 444, 422 or 420");
        }
    }

    image = image.contiguous();
    const auto* pixels = image.data_ptr<unsigned char>();
    const int height = static_cast<int>(image.size(0));
    const int width = static_cast<int>(image.size(1));
    const int pitch = width * channels;
    const int format = channels == 1 ? TJPF_GRAY : TJPF_RGB;
    const int mcuWidth = tjMCUWidth[samp];
    const int mcuHeight = tjMCUHeight[samp];

    // Every strip but the last covers whole MCU rows, so each one's blocks are
    // exactly the blocks of a single-pass encode. A restart interval counts
    // MCUs in a 16-bit field, which caps the strip height on wide images.
    const int mcusPerRow = (width + mcuWidth - 1) / mcuWidth;
    int stripRows = static_cast<int>(kStripPixels / width) / mcuHeight * mcuHeight;
    stripRows = std::clamp(stripRows, mcuHeight, 65535 / mcusPerRow * mcuHeight);
    const int strips = (height + stripRows - 1) / stripRows;
    auto rowsOf = [&](int i) { return std::min(stripRows, height - i * stripRows); };

    std::vector<size_t> offset(strips + 1, 0);
    for (int i = 0; i < strips; ++i) {
        const size_t bound = tj3JPEGBufSize(width, rowsOf(i), samp);
        if (bound == 0)
            throw std::runtime_error("libjpeg-turbo: image is too large");
        offset[i + 1] = offset[i] + bound;
    }
    thread_local Arena arena;
    unsigned char* space = arena.reserve(offset[strips]);
    std::vector<size_t> size(strips, 0);

    auto encodeStrip = [&](int i) {
        tjhandle handle = threadCompressor();
        unsigned char* out = space + offset[i];
        size_t outSize = offset[i + 1] - offset[i];
        if (tj3Set(handle, TJPARAM_QUALITY, quality) || tj3Set(handle, TJPARAM_SUBSAMP, samp) ||
            tj3Set(handle, TJPARAM_NOREALLOC, 1) ||
            tj3Compress8(handle, pixels + size_t(i) * stripRows * pitch, width, pitch, rowsOf(i),
                         format, &out, &outSize))
            throw std::runtime_error(std::string("libjpeg-turbo: ") + tj3GetErrorStr(handle));
        size[i] = outSize;
    };
    StripPool::instance().run(strips, at::get_num_threads(), encodeStrip);

    torch::Tensor result;
    if (strips == 1) {
        result = torch::empty({static_cast<int64_t>(size[0])}, torch::kUInt8);
        std::memcpy(result.data_ptr<unsigned char>(), space, size[0]);
    } else {
        // Strip 0's headers with the full height and a DRI segment, then each
        // strip's entropy-coded data separated by RST0..RST7. A restart resets
        // the DC predictors and byte-aligns the bit stream, which is exactly
        // how every strip's own scan began and ended.
        std::vector<StripLayout> layout(strips);
        size_t total = 6 + 2;  // DRI + EOI
        for (int i = 0; i < strips; ++i) {
            layout[i] = parseStrip(space + offset[i], size[i]);
            total += i == 0 ? size[0] - 2 : 2 + size[i] - layout[i].data - 2;
        }
        result = torch::empty({static_cast<int64_t>(total)}, torch::kUInt8);
        unsigned char* dst = result.data_ptr<unsigned char>();
        const unsigned char* first = space + offset[0];
        std::memcpy(dst, first, layout[0].sos);
        dst[layout[0].sofHeight] = static_cast<unsigned char>(height >> 8);
        dst[layout[0].sofHeight + 1] = static_cast<unsigned char>(height & 0xFF);
        dst += layout[0].sos;
        const int interval = mcusPerRow * (stripRows / mcuHeight);
        const unsigned char dri[6] = {0xFF, 0xDD, 0x00, 0x04, static_cast<unsigned char>(interval >> 8),
                                      static_cast<unsigned char>(interval & 0xFF)};
        std::memcpy(dst, dri, 6);
        dst += 6;
        const size_t head = size[0] - 2 - layout[0].sos;
        std::memcpy(dst, first + layout[0].sos, head);
        dst += head;
        for (int i = 1; i < strips; ++i) {
            *dst++ = 0xFF;
            *dst++ = static_cast<unsigned char>(0xD0 + ((i - 1) & 7));
            const size_t data = size[i] - 2 - layout[i].data;
            std::memcpy(dst, space + offset[i] + layout[i].data, data);
            dst += data;
        }
        *dst++ = 0xFF;
        *dst++ = 0xD9;
    }
    arena.trim();
    return result;
}

} // namespace nelux
