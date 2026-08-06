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
#include <pybind11/pybind11.h>
#include <shared_mutex>
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

    // Check if using hardware encoder. Takes the lifecycle lock: it reads the
    // `encoder` member, which close() swaps out and destroys.
    bool isHardwareEncoder() const;

    // Copy audio and/or subtitle streams from `source` into the output
    // (ffmpeg `-c:a copy -c:s copy`). Must be called before the first
    // encodeFrame. `start`/`end` (seconds) trim the copied streams; end < 0
    // means "to end of source". When `allowTranscode` is true, streams whose
    // codec cannot be copied into the output container are re-encoded to the
    // container default instead of being dropped.
    void addPassthrough(const std::string& source, bool audio, bool subtitles,
                        double start, double end, bool allowTranscode);

  private:
    // Everything below is pipeline internals. It is private so that "every
    // operation on an encoder goes through lifecycleMu_" is checked by the
    // compiler rather than by review: the codec handle, the worker threads, the
    // buffer pools and both lock helpers are unreachable from outside this
    // class, so the public entry points above are the only way in and each of
    // them takes the lock.

    // True for single-plane grayscale output formats that get the verbatim
    // full-range data path (GRAY8 / GRAY16LE / GRAY16BE) instead of the RGB24
    // async convert pipeline. Depth maps and other single-channel data need
    // exact, full-range, up-to-16-bit values — not video-luma range/precision.
    static bool isGrayVerbatimPixfmt(AVPixelFormat f)
    {
        return f == AV_PIX_FMT_GRAY8 || f == AV_PIX_FMT_GRAY16LE ||
               f == AV_PIX_FMT_GRAY16BE;
    }

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
    //
    // All three of the lifecycle helpers below mutate pipeline state that has
    // no mutex of its own (workersStarted, convertStarted, numConvertWorkers,
    // maxFramesInFlight, the encodeThread / convertWorkers thread objects, the
    // buffer pools). The CALLER must hold lifecycleMu_ exclusively. Two threads
    // racing startEncodeWorkersIfNeeded() both saw workersStarted == false and
    // both assigned to the same, by then joinable, std::thread — which is an
    // unconditional std::terminate.
    void startEncodeWorkersIfNeeded();    // spawns the single submit thread
    void ensureConvertPipeline();         // lazily spawns convert pool (CPU path only)
    void stopEncodeWorkers();             // drains all queued frames, then joins

    // The locked bodies of the public entry points. Called through
    // underEncoderLock() so the lock/GIL dance lives in exactly one place.
    // Both run with lifecycleMu_ held exclusively and the GIL dropped.
    void encodeFrameLocked(torch::Tensor& frame);
    void closeLocked();
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
        std::vector<uint8_t>* staging = nullptr;  // recycled packed-RGB buffer
        nelux::Frame* yuv = nullptr;              // recycled YUV frame to fill
        int64_t seq = 0;
        // Layout of `staging`: RGB24 for 8-bit input, RGB48LE when the caller
        // supplied deep-colour data and the destination can store it.
        AVPixelFormat srcFmt = AV_PIX_FMT_RGB24;
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

    // Inner lock: the caller<->worker queue state only (convertQueue, readyMap,
    // the free pools, inFlight, workerError, retiredTensors). Taken by the
    // worker threads, which never touch lifecycleMu_, and by the callers.
    //
    // Lock order is lifecycleMu_ then `mu`, never the reverse. Note that `mu` is
    // NOT always taken under lifecycleMu_: encodeFrame's prologue takes it on
    // its own, before the lifecycle lock, to hand retiredTensors back to the GIL
    // (see encodeFrame). That is allowed — acquiring the inner lock alone
    // violates no ordering — and it is why closeLocked() must take `mu` too when
    // it drains the same deque; lifecycleMu_ alone does not exclude that caller.
    //
    // INVARIANT: `mu` must never be held across a GIL acquisition. It is
    // routinely taken WITH the GIL already held (encodeFrame's prologue), so a
    // thread that held `mu` and then waited for the GIL would close a cycle and
    // deadlock. Anything that needs the GIL — freeing torch tensors is the only
    // case — swaps the data out under `mu`, releases `mu`, and frees after.
    std::mutex mu;
    std::condition_variable cvConvert;  // convert workers wait for convertQueue
    std::condition_variable cvReady;    // submit thread waits for readyMap[nextSubmitSeq]
    std::condition_variable cvFree;     // caller waits for in-flight capacity

    int numConvertWorkers = 0;       // resolved at start (<=0 => auto)
    size_t maxFramesInFlight = 8;    // bounded == backpressure
    std::exception_ptr workerError;  // first error from any worker, re-raised
    bool workersStarted = false;     // submit thread up

    // Serialises EVERY public operation on one encoder, the same contract
    // VideoReader::lifecycleMu_ establishes for the reader: operations on one
    // object are serialised, separate objects run fully in parallel.
    //
    // The GIL used to provide that exclusion for free. It no longer does — the
    // encode path drops it for the convert/submit handoff and close() drops it
    // for the codec drain and trailer write — so it has to be explicit.
    // Without it: two closers both saw a non-null `encoder` and both ran the
    // full drain/mux/av_write_trailer on one AVFormatContext (access
    // violation); two first-time encoders both saw workersStarted == false and
    // both assigned the same joinable std::thread (std::terminate); and an
    // encode racing a close ran the pipeline against an encoder that was being
    // freed underneath it.
    //
    // It is a shared_mutex only so the read-only accessor (isHardwareEncoder)
    // does not serialise against itself. Everything that touches the codec, the
    // worker threads or the pools takes it EXCLUSIVELY, because they all
    // ultimately drive one stateful AVCodecContext.
    //
    // NOT recursive: nothing holding it may call another public entry point.
    // `mu` (above) is the inner lock: it may be taken while holding this one, or
    // on its own, but this one is never taken while holding `mu`.
    mutable std::shared_mutex lifecycleMu_;

    // Run `fn` under the exclusive encoder lock with the GIL dropped.
    //
    // `release` is declared BEFORE the lock guard, so it destructs AFTER it:
    // the lock is dropped while the GIL is still released. Reversed, a thread
    // would sit waiting to re-acquire the GIL while still holding the lock, and
    // close() — which reaches its own lock holding the GIL — would deadlock
    // against it.
    //
    // Code inside `fn` may nonetheless re-acquire the GIL — closeLocked() does,
    // to free retired CUDA tensors. That is safe for one specific reason: no
    // thread ever blocks on lifecycleMu_ while holding the GIL, because EVERY
    // acquisition goes through this helper or its read counterpart and both drop
    // the GIL first. Add a lock site that bypasses them and this stops being
    // true, and the acquire in closeLocked() becomes a deadlock. (VideoReader
    // never re-acquires the GIL under its lock, so this is the one place the
    // encoder depends on that property rather than merely satisfying it.)
    //
    // The PyGILState_Check() guard is required: close() also runs from
    // ~VideoEncoder, which can fire from a non-Python thread or during
    // interpreter finalisation, where releasing a GIL we do not hold is UB.
    template <class Fn> auto underEncoderLock(Fn&& fn) -> decltype(fn())
    {
        if (PyGILState_Check())
        {
            pybind11::gil_scoped_release release;
            std::unique_lock<std::shared_mutex> lk(lifecycleMu_);
            return fn();
        }
        std::unique_lock<std::shared_mutex> lk(lifecycleMu_);
        return fn();
    }

    // Read-only counterpart. Same ordering rules; takes a SHARED lock so
    // concurrent readers do not disturb each other while still excluding the
    // writers (encodeFrame, addPassthrough, close).
    template <class Fn> auto underEncoderLockRead(Fn&& fn) const -> decltype(fn())
    {
        if (PyGILState_Check())
        {
            pybind11::gil_scoped_release release;
            std::shared_lock<std::shared_mutex> lk(lifecycleMu_);
            return fn();
        }
        std::shared_lock<std::shared_mutex> lk(lifecycleMu_);
        return fn();
    }

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

