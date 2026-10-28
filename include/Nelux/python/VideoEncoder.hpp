#pragma once
#ifndef VIDEO_ENCODER_HPP
#define VIDEO_ENCODER_HPP

#include "Encoder.hpp"
#include <cpu/RGBToAuto.hpp>
#include <cpu/RowLoader.hpp>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <exception>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <pybind11/pybind11.h>
#include <shared_mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef NELUX_ENABLE_CUDA
#include <cuda_runtime.h>
#include <gpu/RGBToYUVExactGPU.hpp>
#endif

namespace nelux
{

// Describes a CPU [H, W, 3|4] tensor for the row loader. False when the
// loader does not cover its dtype or strides (the caller then normalises it
// with torch).
bool rowSourceFor(const torch::Tensor& frame, conversion::cpu::RowSource& rows);

class VideoEncoder
{
  public:
    // Constructor with optional arguments including NVENC options.
    // `preset` accepts either an int (uses per-codec mapping table) or a
    // string (forwarded straight to av_dict_set("preset", ...) so callers
    // can pass exact ffmpeg-cli preset names like "medium" or "p4").
    // `extraOptions` are arbitrary AVOption key/value pairs forwarded to
    // avcodec_open2 — applied AFTER built-in options so they override.
    // `resize` opts the encoder into scaling input frames to width x height:
    // encodeFrame then accepts any [H,W,C]/[H,W] input size (locked by the
    // first frame) and the spatial scale folds into the swscale pass the
    // convert pipeline already runs — there is no separate resize stage.
    // `resizeFilter` picks the scaling kernel (same names as the reader's
    // resize_filter / ffmpeg's -sws_flags).
    VideoEncoder(const std::string& filename,
                 std::optional<std::string> codec = std::nullopt,
                 std::optional<int> width = std::nullopt,
                 std::optional<int> height = std::nullopt,
                 std::optional<int> bitRate = std::nullopt,
                 // double, not float: float32 cannot hold 24000/1001 closely
                 // enough for av_d2q to recover the fraction it came from.
                 std::optional<double> fps = std::nullopt,
                 std::optional<int> preset = std::nullopt,
                 std::optional<int> cq = std::nullopt,
                 std::optional<std::string> pixelFormat = std::nullopt,
                 std::optional<std::string> presetStr = std::nullopt,
                 std::map<std::string, std::string> extraOptions = {},
                 bool resize = false,
                 const std::string& resizeFilter = "bilinear",
                 bool memoryOutput = false,
                 // A seekable binary stream the container is written into as
                 // it is muxed (see writeDestination); None = `filename`.
                 pybind11::object destination = pybind11::object(),
                 // An exact frame rate (num, den) in place of `fps`, which
                 // frameRateFromDouble reads as a rate a person typed:
                 // nelux.transcode hands over the source's own rational.
                 std::optional<std::pair<int, int>> exactFrameRate = std::nullopt);

    ~VideoEncoder();

    void encodeFrame(torch::Tensor frame);
    void close();
    torch::Tensor getEncodedData();

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

    // Encode the video of `source` from its decoded frames, without the
    // YUV->RGB->YUV round trip a VideoReader -> encode_frame loop takes
    // (nelux.transcode). Frames already in the output format reach the codec
    // as decoded; others take one swscale pass. `start`/`end` are seconds on
    // the source's timestamps, end < 0 = to the end. Into NVENC, a source
    // NVDEC decodes exactly is decoded on the GPU unless `gpuDecode` is
    // false; everything else decodes on the CPU. Before any encode_frame.
    // Defined in Transcode.cpp. Returns (frames sent as decoded, frames
    // converted).
    std::pair<int64_t, int64_t> transcodeFrom(const std::string& source, double start,
                                              double end, bool gpuDecode = true);

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

    // --- Binary-stream destination -------------------------------------------
    // The muxer writes and seeks through destination.write/seek on whichever
    // thread muxes: the submit thread, or the caller under lifecycleMu_. Each
    // callback takes the GIL, which is deadlock-free because no thread waits
    // for those threads while holding the GIL (underEncoderLock drops it
    // first; see there). Declared before `encoder`, whose teardown can still
    // write. The py::objects are only touched with the GIL held, and
    // ~VideoEncoder drops them under it.
    pybind11::object destination_;
    int64_t destinationBase_ = 0;        // destination.tell() at construction
    pybind11::object destinationError_;  // first exception write/seek raised
    // The thread running destination.write/seek, so a call back into this
    // encoder from there raises instead of deadlocking on lifecycleMu_.
    std::atomic<std::thread::id> destinationThread_{};
    // The thread running Python's signal handlers inside transcodeFrom, which
    // holds lifecycleMu_ meanwhile: a handler calling this encoder raises.
    std::atomic<std::thread::id> signalHandlerThread_{};
    nelux::Encoder::OutputSink outputSink_;
    int writeDestination(const uint8_t* data, int size);
    int64_t seekDestination(int64_t offset, int whence);
    void rejectDestinationReentry() const;
    // Raises the destination's own exception: always for a call that failed
    // (`failed`, it is what made it fail), once for one that succeeded
    // anyway -- the close() of an mp4 whose trailer seek the stream refused;
    // a later close() of the torn-down encoder then has nothing to report.
    void raiseDestinationError(bool failed);
    bool destinationErrorReported_ = false; // under the GIL

    std::unique_ptr<nelux::Encoder> encoder;
    int width, height;
    AVPixelFormat outputPixelFormat;  // Actual pixel format used
    std::string outputPath_;
    bool memoryOutput_ = false;
    nelux::Encoder::MemoryOutput encodedOutput_;
    // The finished memory output, handed out by every get_encoded_data().
    torch::Tensor encodedData_;
    bool directStillImage_ = false;   // one image2 file, no numbered pattern
    bool directStillImageEncoded_ = false;
    bool borrowStillImageInput_ = false; // safe only when codec has one thread
    bool adaptivePngStill_ = false;   // caller left PNG compression unspecified

    // --- Encoder-side resize (opt-in) ----------------------------------------
    // When resizeEnabled_, encodeFrame accepts input of any spatial size and
    // the convert pipeline's sws pass scales it to width x height. The source
    // size is read from the first frame's shape and locked (srcWidth_ == 0
    // means "not locked yet"): every cached SwsContext and staging buffer
    // downstream is built for exactly one source size, so a later frame with a
    // different size is an error, not a silent re-init. Guarded by
    // lifecycleMu_ like the rest of the pipeline state.
    bool resizeEnabled_ = false;
    int resizeFlags_ = SWS_BILINEAR;  // scaling kernel (shared SWS_* mapping)
    int srcWidth_ = 0;
    int srcHeight_ = 0;
    
#ifdef NELUX_ENABLE_CUDA
    // CUDA tensor input: RGB->YUV runs on the GPU with the byte-exact kernels
    // (RGBToYUVExactGPU.hpp), on the caller's stream at encode_frame time,
    // into an encoder-owned device slot. One converter per source layout the
    // caller has sent (rgb24 / rgba / rgb48 / rgba64); `conv` is null when the
    // pair has no kernel, and `matchesCpu` says selfTest() proved it writes
    // the CPU path's bytes on this libswscale.
    struct GpuConverterEntry
    {
        AVPixelFormat srcFmt = AV_PIX_FMT_NONE;
        std::unique_ptr<nelux::backends::cuda::RgbToYuvExactConverter> conv;
        bool tested = false;
        bool matchesCpu = false;
    };
    std::vector<GpuConverterEntry> gpuConverters_;
    // @p stream carries the one-time table upload and self-test.
    nelux::backends::cuda::RgbToYuvExactConverter* gpuConverterFor(AVPixelFormat srcFmt,
                                                                   bool needCpuBytes,
                                                                   cudaStream_t stream);

    // Device YUV frames the GPU convert writes. NVENC entries hold one from
    // encode_frame until the submit thread has copied it into a hw frame
    // (maxFramesInFlight of them); a software encoder alternates between
    // slots 0 and 1 (see swDownloadStream_). One allocation per slot,
    // planes laid out with gpuSlotPitch_ and freed in closeLocked().
    std::vector<uint8_t*> gpuSlots_;
    std::deque<int> freeGpuSlots_;  // guarded by `mu`
    // Per slot, recorded after its convert on the caller's stream (was one
    // created and destroyed per frame). A slot is in flight once at a time
    // and back in freeGpuSlots_ only after the submit thread queued its wait,
    // so a record never lands under a pending wait (one event shared by every
    // frame did, with several caller streams).
    std::vector<cudaEvent_t> gpuSlotReady_;
    int gpuSlotDevice_ = -1;
    int gpuPlanes_ = 0;
    int64_t gpuSlotBytes_ = 0;
    int64_t gpuSlotPitch_[3] = {};
    int64_t gpuPlaneOffset_[3] = {};
    int64_t gpuRowBytes_[3] = {};
    int gpuRows_[3] = {};
    void ensureGpuSlots(int count);
    nelux::backends::cuda::RgbToYuvExactDest gpuSlotDest(int slot) const;
    bool encodeCudaFrame(const torch::Tensor& frame, AVPixelFormat srcFmt, int inW, int inH);
    // NVENC: take a free slot, let @p fill enqueue its convert on @p stream
    // and queue the slot for the submit thread behind an event recorded there.
    void enqueueGpuSlot(cudaStream_t stream,
                        const std::function<void(const nelux::backends::cuda::RgbToYuvExactDest&)>& fill);

    // CPU tensor into NVENC: the RGB goes up to cpuRgbDevice_ on
    // cpuUploadStream_ (nelux's own, so no torch CUDA is needed) and the
    // kernel converts it into a slot there, instead of swscale on the CPU
    // and a YUV upload on the submit thread. cpuUploadDone_ (blocking-sync)
    // holds encode_frame until a pinned tensor has been read. With @p rows
    // the frame is loaded as RGB48/RGBA64 into the next of cpuRgbPinned_
    // (page-locked, reused once its blocking-sync `done` has fired) and goes
    // up from there; into cpuRgbHost_ if page-locked memory cannot be had.
    bool encodeCpuFrameOnGpu(const torch::Tensor& frame,
                             const nelux::conversion::cpu::RowSource* rows, AVPixelFormat srcFmt,
                             int inW, int inH);
    void* cpuRgbDevice_ = nullptr;
    size_t cpuRgbBytes_ = 0;
    std::vector<uint8_t> cpuRgbHost_;
    struct CpuRgbPinned
    {
        uint8_t* host = nullptr;
        size_t size = 0;
        cudaEvent_t done = nullptr;
    };
    CpuRgbPinned* nextCpuRgbPinned(size_t bytes);
    CpuRgbPinned cpuRgbPinned_[2];
    int cpuRgbNext_ = 0;
    bool cpuRgbPinFailed_ = false;
    cudaStream_t cpuUploadStream_ = nullptr;
    cudaEvent_t cpuUploadDone_ = nullptr;
    bool cpuOnGpuFailed_ = false;  // setup failed once: CPU frames stay on the CPU path

    cudaStream_t encoderStream = nullptr;  // submit thread: slot -> hw frame copies
    cudaEvent_t encoderCopyDone = nullptr; // submit thread: blocking-sync wait on that stream

    // CUDA tensor into a software encoder: the convert writes a slot on the
    // caller's stream and swDownloadStream_ copies its YUV planes into the
    // frame's host buffer behind it, so encode_frame does not wait for the
    // GPU. The submit thread sleeps on the frame's blocking-sync event (from
    // freeSwDownloads_) before sending it. The caller's stream waits on
    // swSlotCopied_ before it writes a slot again, so two slots keep the
    // caller's next frame off the copy. All of it is caller-owned
    // (lifecycleMu_) except freeSwDownloads_, which is guarded by `mu`.
    void ensureSwDownload();
    cudaStream_t swDownloadStream_ = nullptr;
    cudaEvent_t swConverted_ = nullptr;     // caller's stream -> download stream
    cudaEvent_t swSlotCopied_[2] = {};      // download stream -> caller's stream
    int swNextSlot_ = 0;
    std::vector<cudaEvent_t> swDownloads_;  // one per frame in flight
    std::deque<cudaEvent_t> freeSwDownloads_;
    // Host buffers for those frames (swHostPool_): page-locked, so the copy
    // is a DMA straight into the frame, up to swPinnedCap_ of them; past that
    // (a codec holding on to more frames, or deep 4K frames over the 256 MiB
    // cap) plain memory, where the copy returns only once it is done, as
    // before.
    static AVBufferRef* allocSwHostBuffer(void* opaque, size_t size);
    size_t swPinnedCap_ = 0;
    size_t swPinnedBuffers_ = 0;

    // CUDA frames the GPU convert does not take come down as RGB into a
    // staging buffer. Page-locked (each buffer registered once), that copy is
    // a DMA the caller sleeps on (stagingDone_, blocking-sync); into pageable
    // memory the driver stages it and spins in torch's context. Caller-owned
    // (lifecycleMu_); unregistered in closeLocked, or before a buffer grows.
    bool pinStaging(std::vector<uint8_t>* staging, int device);
    void unpinStaging(std::vector<uint8_t>* staging);
    std::map<std::vector<uint8_t>*, void*> pinnedStaging_;
    cudaEvent_t stagingDone_ = nullptr;
    int stagingDoneDevice_ = -1;
    bool stagingPinFailed_ = false; // pageable from then on

    // CPU-converted frames into NVENC on the shared context go up from the
    // pool frames themselves, registered page-locked once (pinYuvPool), not
    // through Encoder's staged copy. `uploaded` (blocking-sync) is recorded
    // behind a frame's DMA; prepareYuvForWrite waits on it before the frame
    // is written again. Built before the pool's first use and read-only after
    // that, so every thread reads it without a lock; released in closeLocked.
    struct PinnedYuv
    {
        const AVBufferRef* buffer = nullptr; // the registration, while buf[0] is it
        cudaEvent_t uploaded = nullptr;
    };
    void pinYuvPool();
    std::map<const AVFrame*, PinnedYuv> pinnedYuv_;
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
    void ensureFramePools();              // staging + YUV frame pools (no threads)
    void ensureConvertPipeline();         // lazily spawns convert pool (CPU path only)
    void stopEncodeWorkers();             // drains all queued frames, then joins

    // The locked bodies of the public entry points. Called through
    // underEncoderLock() so the lock/GIL dance lives in exactly one place.
    // Both run with lifecycleMu_ held exclusively and the GIL dropped.
    void encodeFrameLocked(torch::Tensor& frame);
    void encodeStillImageDirect(torch::Tensor& frame, AVPixelFormat srcFmt,
                                int inW, int inH, int srcChannels, bool deep);
    void closeLocked();
    void convertWorkerLoop(int workerId);
    void encodeSubmitLoop();
    void submitYuvFrame(nelux::Frame* yuv);  // submit thread: make-writable + send + drain
    void prepareYuvForWrite(AVFrame* yuv);   // recycled pool frame -> writable, NOPTS

    // Synchronous verbatim grayscale encode (no convert pool, no RGB24 8-bit
    // bottleneck, no range squeeze). Fills a GRAY8/GRAY16 plane directly from a
    // single-channel tensor (8- or 16-bit), tags full range, and sends it to the
    // codec in-line. Used for gray-output pixfmts so single-channel data (depth
    // maps) round-trips exactly. RGB input to a gray-output encoder is converted
    // to luma via grayRgbConverter_.
    // inW/inH are the input frame's dimensions (== width/height unless resize
    // is enabled); grayInput says whether the caller passed a single-channel
    // frame (decided from the shape by encodeFrameLocked, which knows the
    // resize mode).
    void encodeGrayVerbatim(torch::Tensor frame, int inW, int inH, bool grayInput);
    std::unique_ptr<nelux::conversion::cpu::RGBToAutoConverter> grayRgbConverter_;
#ifdef NELUX_ENABLE_CUDA
    // GPU: wait for the slot's convert, copy it into a CUDA AVFrame, send.
    void submitGpuToEncoder(int gpuSlot);
#endif

    // Large CPU frames an AVX2 kernel covers: the caller and the convert
    // workers convert row bands of the caller's tensor straight into a pooled
    // YUV frame, and encode_frame returns once every band is written. Returns
    // false (nothing taken) for every other frame, which stages as before.
    // `rows` is set for a frame the row loader reads (into a scratch buffer a
    // few rows at a time); otherwise `frame` is contiguous in `srcFmt`.
    bool encodeBandedFrame(const torch::Tensor& frame,
                           const nelux::conversion::cpu::RowSource* rows,
                           AVPixelFormat srcFmt);
    // Caller-owned (lifecycleMu_): yields the plan the band jobs share.
    std::unique_ptr<nelux::conversion::cpu::RGBToAutoConverter> bandConverter_;
    // Row loader scratch, one per convert worker plus one for the caller
    // (the last); sized when the convert pool starts.
    std::vector<std::vector<uint8_t>> loadScratch_;

    // RGB frame awaiting conversion (CPU path), assigned a target YUV frame.
    struct ConvertJob
    {
        std::vector<uint8_t>* staging = nullptr;  // recycled packed-RGB buffer
        nelux::Frame* yuv = nullptr;              // recycled YUV frame to fill
        int64_t seq = 0;
        // Layout of `staging`: RGB24 for 8-bit input, RGB48LE when the caller
        // supplied deep-colour data and the destination can store it.
        AVPixelFormat srcFmt = AV_PIX_FMT_RGB24;

        // Band job (staging == nullptr): rows [y0, y1) of the caller's
        // tensor, packed at `bandSrc` or read through `bandRows`, into `yuv`
        // with `kernel`. The caller waits on cvBands until bandsPending reaches
        // zero, so all of them stay alive.
        nelux::conversion::cpu::FastKernel kernel;
        const uint8_t* bandSrc = nullptr;
        const nelux::conversion::cpu::RowSource* bandRows = nullptr;
        int y0 = 0;
        int y1 = 0;
    };

    // A frame ready for the submit thread, in `readyMap` keyed by seq.
    struct ReadyEntry
    {
        nelux::Frame* yuv = nullptr;   // CPU path: already converted, ready to send
#ifdef NELUX_ENABLE_CUDA
        int gpuSlot = -1;              // GPU path: device YUV, converted at encode_frame
        bool isGpu = false;
        cudaEvent_t downloadDone = nullptr;  // software encoder: `yuv` still arriving from the GPU
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
    // Fresh buffers for yuvPool frames the codec still references (see
    // convertWorkerLoop). Uninit only marks it for freeing: buffers the codec
    // holds past this encoder's teardown still return to it safely.
    struct BufferPoolUninit
    {
        void operator()(AVBufferPool* p) const { av_buffer_pool_uninit(&p); }
    };
    std::unique_ptr<AVBufferPool, BufferPoolUninit> yuvBufferPool;
#ifdef NELUX_ENABLE_CUDA
    std::unique_ptr<AVBufferPool, BufferPoolUninit> swHostPool_;  // see allocSwHostBuffer
#endif

    std::deque<ConvertJob> convertQueue;     // RGB awaiting convert (CPU path)
    std::map<int64_t, ReadyEntry> readyMap;  // seq -> ready frame, drained in order
    int64_t enqueueSeq = 0;     // caller-assigned, monotonic
    int64_t nextSubmitSeq = 0;  // submit thread cursor
    int64_t inFlight = 0;       // admitted but not yet submitted (backpressure)

    // Inner lock: the caller<->worker queue state only (convertQueue, readyMap,
    // the free pools, inFlight, workerError). Taken by the worker threads,
    // which never touch lifecycleMu_, and by the callers under lifecycleMu_.
    // Lock order is lifecycleMu_ then `mu`, never the reverse.
    //
    // INVARIANT: `mu` must never be held across a GIL acquisition, nor while
    // destroying anything whose destructor may take the GIL (a
    // pybind11::error_already_set in workerError): swap the data out under
    // `mu`, release it, and destroy after.
    std::mutex mu;
    std::condition_variable cvConvert;  // convert workers wait for convertQueue
    std::condition_variable cvReady;    // submit thread waits for readyMap[nextSubmitSeq]
    std::condition_variable cvFree;     // caller waits for in-flight capacity
    std::condition_variable cvBands;    // caller waits for its band jobs
    int bandsPending = 0;               // band jobs of the current frame not yet done
    std::exception_ptr bandError;       // first failure among them

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
    // Code inside `fn` could therefore re-acquire the GIL safely: no thread
    // ever blocks on lifecycleMu_ while holding the GIL, because EVERY
    // acquisition goes through this helper or its read counterpart and both
    // drop the GIL first. Add a lock site that bypasses them and that stops
    // being true.
    //
    // The PyGILState_Check() guard is required: close() also runs from
    // ~VideoEncoder, which can fire from a non-Python thread or during
    // interpreter finalisation, where releasing a GIL we do not hold is UB.
    //
    // A failure the destination stream caused is raised as the stream's own
    // exception, after the lock is dropped -- also when `fn` succeeded
    // anyway (once): movenc and the avi muxer ignore a failed seek while
    // writing the trailer, so close() would return with headers unpatched.
    template <class Fn> auto underEncoderLock(Fn&& fn) -> decltype(fn())
    {
        rejectDestinationReentry();
        auto locked = [&]() -> decltype(fn())
        {
            try
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
            catch (...)
            {
                raiseDestinationError(true);
                throw;
            }
        };
        if constexpr (std::is_void_v<decltype(fn())>)
        {
            locked();
            raiseDestinationError(false);
        }
        else
        {
            auto result = locked();
            raiseDestinationError(false);
            return result;
        }
    }

    // Read-only counterpart. Same ordering rules; takes a SHARED lock so
    // concurrent readers do not disturb each other while still excluding the
    // writers (encodeFrame, addPassthrough, close).
    template <class Fn> auto underEncoderLockRead(Fn&& fn) const -> decltype(fn())
    {
        rejectDestinationReentry();
        if (PyGILState_Check())
        {
            pybind11::gil_scoped_release release;
            std::shared_lock<std::shared_mutex> lk(lifecycleMu_);
            return fn();
        }
        std::shared_lock<std::shared_mutex> lk(lifecycleMu_);
        return fn();
    }

    bool framePoolsReady_ = false;   // ensureFramePools ran
    bool convertStarted = false;     // convert pool up (CPU path only)
    bool stopping = false;

    // A converter for this encoder's output, built the way every CPU convert
    // builds it (convert workers, band jobs, a still image), so the verdict
    // the constructor's probe keeps is the one they look up.
    std::unique_ptr<nelux::conversion::cpu::RGBToAutoConverter> makeConverter() const;

    nelux::Encoder::EncodingProperties inferEncodingProperties(
        const std::string& filename, std::optional<std::string> codec,
        std::optional<int> width, std::optional<int> height, std::optional<int> bitRate,
        std::optional<double> fps,
        std::optional<int> preset, std::optional<int> cq,
        std::optional<std::string> pixelFormat,
        std::optional<std::string> presetStr,
        std::map<std::string, std::string> extraOptions);

    // The AVX2 kernel's exactness probe, started by the constructor so it
    // runs before the first frame needs it. It owns its converter and
    // touches nothing of this object. Joined in closeLocked and again on
    // destruction, so no path leaves it joinable (std::terminate). Declared
    // last: destroyed first. Not std::jthread, which the libc++ of the macOS
    // wheel build lacks.
    struct JoiningThread
    {
        std::thread thread;
        ~JoiningThread() { join(); }
        void join()
        {
            if (thread.joinable())
                thread.join();
        }
    };
    JoiningThread verdictProbe_;
};

} // namespace nelux

#endif // VIDEO_ENCODER_HPP

