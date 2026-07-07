#pragma once
#ifndef VIDEO_ENCODER_HPP
#define VIDEO_ENCODER_HPP

#include "Encoder.hpp"
#include <cpu/RGBToAuto.hpp>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifdef NELUX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <gpu/RGBToAutoGPU.hpp>
#endif

namespace nelux
{

class VideoEncoder
{
  public:
    // Constructor with optional arguments including NVENC options.
    // `preset` accepts either an int (uses per-codec mapping table) or a
    // string (forwarded straight to av_dict_set("preset", ...) so callers
    // can pass exact ffmpeg-cli preset names like "medium" or "p4").
    // `extraOptions` are arbitrary AVOption key/value pairs forwarded to
    // avcodec_open2 — applied AFTER built-in options so they override.
    VideoEncoder(const std::string& filename,
                 std::optional<std::string> codec = std::nullopt,
                 std::optional<int> width = std::nullopt,
                 std::optional<int> height = std::nullopt,
                 std::optional<int> bitRate = std::nullopt,
                 std::optional<float> fps = std::nullopt,
                 std::optional<int> preset = std::nullopt,
                 std::optional<int> cq = std::nullopt,
                 std::optional<std::string> pixelFormat = std::nullopt,
                 std::optional<std::string> presetStr = std::nullopt,
                 std::map<std::string, std::string> extraOptions = {});

    ~VideoEncoder();

    void encodeFrame(torch::Tensor frame);
    void close();

    // True for single-plane grayscale output formats that get the verbatim
    // full-range data path (GRAY8 / GRAY16LE / GRAY16BE) instead of the RGB24
    // async convert pipeline. Depth maps and other single-channel data need
    // exact, full-range, up-to-16-bit values — not video-luma range/precision.
    static bool isGrayVerbatimPixfmt(AVPixelFormat f)
    {
        return f == AV_PIX_FMT_GRAY8 || f == AV_PIX_FMT_GRAY16LE ||
               f == AV_PIX_FMT_GRAY16BE;
    }

    // Copy audio and/or subtitle streams from `source` into the output
    // (ffmpeg `-c:a copy -c:s copy`). Must be called before the first
    // encodeFrame. `start`/`end` (seconds) trim the copied streams; end < 0
    // means "to end of source". When `allowTranscode` is true, streams whose
    // codec cannot be copied into the output container are re-encoded to the
    // container default instead of being dropped.
    void addPassthrough(const std::string& source, bool audio, bool subtitles,
                        double start, double end, bool allowTranscode);
    
    // Check if using hardware encoder
    bool isHardwareEncoder() const { return encoder && encoder->isHardwareEncoder(); }
    
    nelux::Encoder::EncodingProperties props;

    std::unique_ptr<nelux::Encoder> encoder;
    int width, height;
    AVPixelFormat outputPixelFormat;  // Actual pixel format used
    
#ifdef NELUX_ENABLE_CUDA
    // GPU converter for zero-copy encoding when tensor is on CUDA
    std::unique_ptr<nelux::conversion::gpu::RGBToAutoGPUConverter> gpuConverter;
    cudaStream_t encoderStream = nullptr;
#endif
    
    // Reusable CPU frame to avoid allocation churn (owned by the encode worker
    // once async encoding starts; only ever touched by one thread at a time).
    nelux::Frame cpuFrame;

    // --- Async encode pipeline (fan-out convert -> in-order submit) ----------
    // RGB->YUV swscale is single-threaded and dominates the worker (~80% at 4K),
    // so it is fanned out across K convert workers (mirrors the decoder's
    // convert pool). Each converts a frame in parallel, tags it with a sequence
    // number, and drops it in a reorder map. ONE encode thread pulls frames in
    // sequence order and calls avcodec_send_frame (x264 is a single stateful
    // context -> submit must stay sequential). GPU/NVENC input skips the convert
    // workers: the convert is cheap on the GPU, so those jobs go straight to the
    // submit thread (which does the GPU convert there).
    void startEncodeWorkersIfNeeded();    // spawns the single submit thread
    void ensureConvertPipeline();         // lazily spawns convert pool (CPU path only)
    void stopEncodeWorkers();             // drains all queued frames, then joins
    void convertWorkerLoop(int workerId);
    void encodeSubmitLoop();
    void submitYuvFrame(nelux::Frame* yuv);  // submit thread: make-writable + send + drain

    // Synchronous verbatim grayscale encode (no convert pool, no RGB24 8-bit
    // bottleneck, no range squeeze). Fills a GRAY8/GRAY16 plane directly from a
    // single-channel tensor (8- or 16-bit), tags full range, and sends it to the
    // codec in-line. Used for gray-output pixfmts so single-channel data (depth
    // maps) round-trips exactly. RGB input to a gray-output encoder is converted
    // to luma via grayRgbConverter_.
    void encodeGrayVerbatim(torch::Tensor frame);
    std::unique_ptr<nelux::conversion::cpu::RGBToAutoConverter> grayRgbConverter_;
#ifdef NELUX_ENABLE_CUDA
    // GPU: wait input-ready, RGB->NV12 on stream, copy into CUDA AVFrame, send.
    void submitGpuToEncoder(torch::Tensor& gpuTensor, cudaEvent_t readyEvent);
#endif

    // RGB frame awaiting conversion (CPU path), assigned a target YUV frame.
    struct ConvertJob
    {
        std::vector<uint8_t>* staging = nullptr;  // recycled RGB24 buffer
        nelux::Frame* yuv = nullptr;              // recycled YUV frame to fill
        int64_t seq = 0;
    };

    // A frame ready for the submit thread, in `readyMap` keyed by seq.
    struct ReadyEntry
    {
        nelux::Frame* yuv = nullptr;   // CPU path: already converted, ready to send
#ifdef NELUX_ENABLE_CUDA
        torch::Tensor gpuTensor;       // GPU path: converted on the submit thread
        cudaEvent_t readyEvent = nullptr;
        bool isGpu = false;
#endif
    };

    std::vector<std::thread> convertWorkers;
    std::thread encodeThread;
    // One converter (own SwsContext — not thread-safe to share) per convert worker.
    std::vector<std::unique_ptr<nelux::conversion::cpu::RGBToAutoConverter>> converters;

    // Recyclable buffers / frames (sized to cover max frames in flight).
    std::vector<std::unique_ptr<std::vector<uint8_t>>> stagingPool;
    std::deque<std::vector<uint8_t>*> freeStaging;
    std::vector<std::unique_ptr<nelux::Frame>> yuvPool;
    std::deque<nelux::Frame*> freeYuv;

    std::deque<ConvertJob> convertQueue;     // RGB awaiting convert (CPU path)
    std::map<int64_t, ReadyEntry> readyMap;  // seq -> ready frame, drained in order
#ifdef NELUX_ENABLE_CUDA
    // GPU tensors the submit thread is done with. The submit thread only MOVES
    // them here (no refcount op, no GIL); the caller frees them at encode_frame
    // entry while it still holds the GIL. This avoids the submit thread taking
    // the GIL per frame, which stalled it behind the GIL-heavy host pipeline.
    std::deque<torch::Tensor> retiredTensors;
#endif
    int64_t enqueueSeq = 0;     // caller-assigned, monotonic
    int64_t nextSubmitSeq = 0;  // submit thread cursor
    int64_t inFlight = 0;       // admitted but not yet submitted (backpressure)

    std::mutex mu;
    std::condition_variable cvConvert;  // convert workers wait for convertQueue
    std::condition_variable cvReady;    // submit thread waits for readyMap[nextSubmitSeq]
    std::condition_variable cvFree;     // caller waits for in-flight capacity

    int numConvertWorkers = 0;       // resolved at start (<=0 => auto)
    size_t maxFramesInFlight = 8;    // bounded == backpressure
    std::exception_ptr workerError;  // first error from any worker, re-raised
    bool workersStarted = false;     // submit thread up
    bool convertStarted = false;     // convert pool up (CPU path only)
    bool stopping = false;

    nelux::Encoder::EncodingProperties inferEncodingProperties(
        const std::string& filename, std::optional<std::string> codec,
        std::optional<int> width, std::optional<int> height, std::optional<int> bitRate,
        std::optional<float> fps,
        std::optional<int> preset, std::optional<int> cq,
        std::optional<std::string> pixelFormat,
        std::optional<std::string> presetStr,
        std::map<std::string, std::string> extraOptions);
};

} // namespace nelux

#endif // VIDEO_ENCODER_HPP

