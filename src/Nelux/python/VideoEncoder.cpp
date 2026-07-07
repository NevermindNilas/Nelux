#include "python/VideoEncoder.hpp"
#include <cctype>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <cpu/RGBToAuto.hpp>

#ifdef NELUX_ENABLE_CUDA
#include <CudaStream.hpp>  // nelux::currentCudaStream (no eager c10_cuda link)
#endif

extern "C"
{
#include <libavutil/pixdesc.h>
}

namespace fs = std::filesystem;

namespace nelux
{
    //NOTE --- USED HWC
VideoEncoder::VideoEncoder(const std::string& filename,
                           std::optional<std::string> codec, std::optional<int> width,
                           std::optional<int> height, std::optional<int> bitRate,
                           std::optional<float> fps,
                           std::optional<int> preset,
                           std::optional<int> cq,
                           std::optional<std::string> pixelFormat,
                           std::optional<std::string> presetStr,
                           std::map<std::string, std::string> extraOptions)
{
    auto properties = inferEncodingProperties(filename, codec, width, height, bitRate,
                                              fps, preset, cq, pixelFormat,
                                              presetStr, std::move(extraOptions));
    this->props = properties;
    this->width = properties.width;
    this->height = properties.height;
    this->outputPixelFormat = properties.pixelFormat;

    encoder = std::make_unique<nelux::Encoder>(filename, properties);
    
    // After encoder init, check if NVENC changed the pixel format (e.g., to NV12)
    this->outputPixelFormat = encoder->Properties().pixelFormat;
}

nelux::Encoder::EncodingProperties VideoEncoder::inferEncodingProperties(
    const std::string& filename, std::optional<std::string> codec,
    std::optional<int> width, std::optional<int> height, std::optional<int> bitRate,
    std::optional<float> fps,
    std::optional<int> preset, std::optional<int> cq,
    std::optional<std::string> pixelFormat,
    std::optional<std::string> presetStr,
    std::map<std::string, std::string> extraOptions)
{
    // Populate video encoding settings
    nelux::Encoder::EncodingProperties props;
    props.codec = codec.value_or("h264_mf");
    props.width = width.value_or(1920);
    props.height = height.value_or(1080);
    props.bitRate = bitRate.value_or(4000000); // 4 Mbps default
    // Round to an integer fps and clamp to >= 1. A zero/negative value would
    // produce an invalid time_base {1,0} (division by zero downstream / muxer
    // rejection).
    {
        int roundedFps = static_cast<int>(std::round(fps.value_or(30.0f)));
        props.fps = roundedFps < 1 ? 1 : roundedFps;
    }
    props.gopSize = 60;
    props.maxBFrames = 2;

    // Parse pixel format string
    if (pixelFormat.has_value())
    {
        AVPixelFormat fmt = av_get_pix_fmt(pixelFormat->c_str());
        if (fmt != AV_PIX_FMT_NONE)
        {
            props.pixelFormat = fmt;
        }
        else
        {
            props.pixelFormat = AV_PIX_FMT_YUV420P;
        }
    }
    else
    {
        props.pixelFormat = AV_PIX_FMT_YUV420P;
    }

    // NVENC-specific options
    props.preset = preset.value_or(-1);  // -1 means use default
    props.cq = cq.value_or(-1);          // -1 means use bitrate mode

    // String preset (preferred when set) + arbitrary AVOptions
    if (presetStr.has_value())
        props.presetStr = *presetStr;
    props.extraOptions = std::move(extraOptions);

    // Auto-pick colorspace from resolution. Mirrors decoder convention
    // (AutoToRGB: height>576 => BT.709, else BT.601).
    if (props.colorspace == AVCOL_SPC_UNSPECIFIED)
    {
        if (props.height > 576)
        {
            props.colorspace = AVCOL_SPC_BT709;
            props.colorPrimaries = AVCOL_PRI_BT709;
            props.colorTrc = AVCOL_TRC_BT709;
        }
        else
        {
            props.colorspace = AVCOL_SPC_BT470BG;
            props.colorPrimaries = AVCOL_PRI_BT470BG;
            props.colorTrc = AVCOL_TRC_BT709;
        }
    }

    // Range follows pixfmt. YUVJ* implies full range (JPEG); rest is limited.
    if (props.colorRange == AVCOL_RANGE_UNSPECIFIED)
    {
        const bool isFullRangePixFmt =
            (props.pixelFormat == AV_PIX_FMT_YUVJ420P ||
             props.pixelFormat == AV_PIX_FMT_YUVJ422P ||
             props.pixelFormat == AV_PIX_FMT_YUVJ444P ||
             props.pixelFormat == AV_PIX_FMT_YUVJ440P ||
             props.pixelFormat == AV_PIX_FMT_YUVJ411P);
        props.colorRange = isFullRangePixFmt ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
    }

    return props;
}

void VideoEncoder::addPassthrough(const std::string& source, bool audio,
                                  bool subtitles, double start, double end,
                                  bool allowTranscode)
{
    if (!encoder)
        throw std::runtime_error("Encoder is not initialized");
    // The container header is written on the first encodeFrame; copied streams
    // must be registered before that. workersStarted flips on the first frame.
    if (workersStarted)
        throw std::runtime_error(
            "add_passthrough must be called before the first encode_frame");
    encoder->addInputStreams(source, audio, subtitles, start, end,
                             allowTranscode);
}

void VideoEncoder::encodeFrame(torch::Tensor frame)
{
    if (!encoder)
        throw std::runtime_error("Encoder is not initialized");

    // Validate input size up front, while the GIL is still held so this raises
    // as a clean Python exception. The convert pipeline consumes exactly
    // width*height*3 RGB bytes; accept either a 3-channel HWC RGB frame or a
    // single-channel grayscale frame (H*W, shape H×W or H×W×1). A grayscale
    // frame is replicated to RGB below so the rest of the pipeline is unchanged
    // (R==G==B => the encoder sees neutral chroma / exact luma).
    const int64_t rgbElems = static_cast<int64_t>(width) * height * 3;
    const int64_t grayElems = static_cast<int64_t>(width) * height;
    const bool grayInput = (frame.numel() == grayElems);
    if (frame.numel() != rgbElems && !grayInput)
        throw std::invalid_argument(
            "encode_frame: tensor has " + std::to_string(frame.numel()) +
            " elements, expected " + std::to_string(rgbElems) + " (" +
            std::to_string(height) + "x" + std::to_string(width) +
            "x3 HWC RGB) or " + std::to_string(grayElems) + " (" +
            std::to_string(height) + "x" + std::to_string(width) +
            " grayscale)");

#ifdef NELUX_ENABLE_CUDA
    // Free GPU tensors retired by the submit thread, here while pybind still
    // holds the GIL (torch CUDA dealloc can need it). Moving this off the submit
    // thread avoids a per-frame GIL acquire that otherwise stalls it behind the
    // GIL-heavy host pipeline (TensorRT) and tanks NVENC throughput.
    {
        std::deque<torch::Tensor> toFree;
        {
            std::lock_guard<std::mutex> lk(mu);
            toFree.swap(retiredTensors);
        }
        toFree.clear();  // destruct under the GIL
    }
#endif

    py::gil_scoped_release release;

    // Grayscale input: replicate the single luma channel into a fresh 3-channel
    // RGB tensor so the rest of the pipeline is unchanged (R==G==B). Uses only
    // .to()/.contiguous()/torch::empty plus a raw byte fill (no torch reshape/
    // expand/repeat) to keep the interleave explicit and cheap. The result is a
    // CPU tensor, so a grayscale frame handed to an NVENC encoder takes the
    // CPU-stage -> hwupload path instead of the zero-copy GPU path; RGB input
    // keeps the fast path.
    if (grayInput)
    {
        torch::Tensor g = frame;
        if (g.device().is_cuda())
            g = g.to(torch::kCPU);
        if (g.dtype() == torch::kFloat16 || g.dtype() == torch::kFloat32)
            g = (g.to(torch::kFloat32) * 255.0f).clamp(0, 255).to(torch::kUInt8);
        else if (g.scalar_type() == torch::ScalarType::UInt16)
            g = (g.to(torch::kFloat32) / 257.0f).clamp(0, 255).to(torch::kUInt8);
        else if (g.dtype() != torch::kUInt8)
            g = g.to(torch::kUInt8);
        if (!g.is_contiguous())
            g = g.contiguous();

        torch::Tensor rgb = torch::empty(
            {height, width, 3},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        const uint8_t* src = g.data_ptr<uint8_t>();
        uint8_t* dst = rgb.data_ptr<uint8_t>();
        const int64_t n = static_cast<int64_t>(width) * height;
        for (int64_t i = 0; i < n; ++i)
        {
            const uint8_t v = src[i];
            dst[3 * i + 0] = v;
            dst[3 * i + 1] = v;
            dst[3 * i + 2] = v;
        }
        frame = rgb;
    }

#ifdef NELUX_ENABLE_CUDA
    // GPU path: tensor already on CUDA and the encoder is NVENC -> keep the
    // whole pipeline on the GPU (zero-copy). The caller only does the GPU dtype
    // normalize here, records an event so the worker's stream waits until that
    // data is ready, then hands the tensor to the encode worker and returns.
    // The RGB->NV12 convert, device copy, stream sync and avcodec_send_frame all
    // run on the worker, overlapped with the caller's next decode/inference.
    if (frame.device().is_cuda() && encoder->isHardwareEncoder())
    {
        int deviceIndex = frame.device().index();
        if (deviceIndex < 0)
            deviceIndex = 0;

        cudaError_t deviceErr = cudaSetDevice(deviceIndex);
        if (deviceErr != cudaSuccess)
        {
            throw std::runtime_error("Failed to select CUDA device for NVENC encode: " +
                                     std::string(cudaGetErrorString(deviceErr)));
        }

        // Convert tensor dtype to uint8 if needed (on GPU, on torch's stream).
        if (frame.dtype() == torch::kFloat16 || frame.dtype() == torch::kFloat32)
        {
            frame = (frame.to(torch::kFloat32) * 255.0f).clamp(0, 255).to(torch::kUInt8);
        }
        else if (frame.scalar_type() == torch::ScalarType::UInt16)
        {
            frame = (frame.to(torch::kFloat32) / 257.0f).clamp(0, 255).to(torch::kUInt8);
        }
        else if (frame.dtype() != torch::kUInt8)
        {
            frame = frame.to(torch::kUInt8);
        }
        if (!frame.is_contiguous())
            frame = frame.contiguous();

        // Record an event on the tensor's producer stream (torch current stream,
        // which is where the decode + the dtype ops above ran). The worker makes
        // its encode stream wait on this so the convert never reads stale data.
        cudaEvent_t readyEvent = nullptr;
        if (cudaEventCreateWithFlags(&readyEvent, cudaEventDisableTiming) != cudaSuccess)
            readyEvent = nullptr;
        if (readyEvent)
        {
            cudaStream_t producerStream = nelux::currentCudaStream(deviceIndex);
            cudaEventRecord(readyEvent, producerStream);
        }

        startEncodeWorkersIfNeeded();

        // GPU jobs skip the convert workers (convert is cheap on the GPU) and go
        // straight into the reorder map for the submit thread to convert + send.
        {
            std::unique_lock<std::mutex> lk(mu);
            cvFree.wait(lk, [&] {
                return inFlight < static_cast<int64_t>(maxFramesInFlight) || workerError;
            });
            if (workerError)
            {
                if (readyEvent) cudaEventDestroy(readyEvent);
                std::exception_ptr e = workerError;
                std::rethrow_exception(e);
            }
            ReadyEntry entry;
            entry.isGpu = true;
            entry.gpuTensor = frame;     // holds CUDA storage alive until encoded
            entry.readyEvent = readyEvent;
            readyMap.emplace(enqueueSeq++, std::move(entry));
            ++inFlight;
        }
        cvReady.notify_one();
        return;
    }
#endif

    // CPU path (fallback for non-CUDA tensors or software encoders)

    // Normalize dtype to uint8 FIRST, on whatever device the tensor is on.
    // Doing this before any download means a CUDA float tensor is reduced to
    // uint8 on the GPU and we transfer 1 byte/channel instead of 2/4 — at 4K
    // that's a 24.8 MB D2H instead of ~99 MB for float32.
    if (frame.dtype() == torch::kFloat16 || frame.dtype() == torch::kFloat32)
    {
        frame = (frame.to(torch::kFloat32) * 255.0f).clamp(0, 255).to(torch::kUInt8);
    }
    else if (frame.scalar_type() == torch::ScalarType::UInt16)
    {
        frame = (frame.to(torch::kFloat32) / 257.0f).clamp(0, 255).to(torch::kUInt8);
    }
    else if (frame.dtype() == torch::kInt16 || frame.dtype() == torch::kInt32)
    {
        frame = frame.to(torch::kFloat32).clamp(0, 255).to(torch::kUInt8);
    }
    else if (frame.dtype() == torch::kInt64)
    {
        frame = frame.clamp(0, 255).to(torch::kUInt8);
    }
    else if (frame.dtype() != torch::kUInt8)
    {
        frame = frame.to(torch::kUInt8);
    }

    if (!frame.is_contiguous())
    {
        frame = frame.contiguous();
    }

    // Hand the frame to the fan-out convert pipeline. The caller pays for ONE
    // copy of the RGB bytes into a recycled staging buffer; a pool of convert
    // workers does the (single-threaded, ~80%-of-the-cost) swscale RGB->YUV in
    // parallel, and one submit thread sends frames to x264 in sequence order.
    // The staging buffer is plain host memory -> no GIL needed on the workers.
    startEncodeWorkersIfNeeded();
    ensureConvertPipeline();   // CPU path needs the swscale convert pool

    const size_t rgbBytes = static_cast<size_t>(width) * height * 3;

    std::vector<uint8_t>* staging = nullptr;
    nelux::Frame* yuv = nullptr;
    int64_t seq = 0;
    {
        std::unique_lock<std::mutex> lk(mu);
        // Backpressure: block until there's an in-flight slot AND a free staging
        // buffer AND a free YUV frame.
        cvFree.wait(lk, [&] {
            return (inFlight < static_cast<int64_t>(maxFramesInFlight) &&
                    !freeStaging.empty() && !freeYuv.empty()) ||
                   workerError;
        });
        if (workerError)
        {
            std::exception_ptr e = workerError;
            std::rethrow_exception(e);
        }
        staging = freeStaging.front(); freeStaging.pop_front();
        yuv = freeYuv.front(); freeYuv.pop_front();
        seq = enqueueSeq++;
        ++inFlight;
    }

    staging->resize(rgbBytes);  // no-op after the first frame

    auto recycleOnError = [&]() {
        std::lock_guard<std::mutex> lk(mu);
        freeStaging.push_back(staging);
        freeYuv.push_back(yuv);
        --inFlight;
        cvFree.notify_all();
    };

    // Copy straight into the staging buffer. For a CUDA tensor, do a single
    // stream-ordered D2H into staging (no intermediate torch CPU tensor).
    if (frame.device().is_cuda())
    {
#ifdef NELUX_ENABLE_CUDA
        int dev = frame.device().index();
        if (dev < 0) dev = 0;
        cudaStream_t s = nelux::currentCudaStream(dev);
        cudaError_t cerr = cudaMemcpyAsync(staging->data(), frame.data_ptr<uint8_t>(),
                                           rgbBytes, cudaMemcpyDeviceToHost, s);
        if (cerr == cudaSuccess)
            cerr = cudaStreamSynchronize(s);
        if (cerr != cudaSuccess)
        {
            recycleOnError();
            throw std::runtime_error("D2H copy to staging failed: " +
                                     std::string(cudaGetErrorString(cerr)));
        }
#else
        frame = frame.to(torch::kCPU);
        std::memcpy(staging->data(), frame.data_ptr<uint8_t>(), rgbBytes);
#endif
    }
    else
    {
        std::memcpy(staging->data(), frame.data_ptr<uint8_t>(), rgbBytes);
    }

    {
        std::lock_guard<std::mutex> lk(mu);
        convertQueue.push_back(ConvertJob{staging, yuv, seq});
    }
    cvConvert.notify_one();
}

void VideoEncoder::startEncodeWorkersIfNeeded()
{
    if (workersStarted)
        return;

    // Resolve convert worker count (used as backpressure depth even for the GPU
    // path). Default: ~half the cores, clamped [2,6].
    if (numConvertWorkers <= 0)
    {
        unsigned hc = std::thread::hardware_concurrency();
        int k = (hc > 0) ? static_cast<int>(hc) / 2 : 4;
        numConvertWorkers = std::clamp(k, 2, 6);
    }
    // workers + 2 keeps every convert worker fed (one filling, one draining)
    // without over-buffering. At 4K each in-flight slot costs ~24.8 MB RGB +
    // 12.4 MB YUV, so the depth directly sets the pool RAM footprint.
    maxFramesInFlight = static_cast<size_t>(numConvertWorkers) + 2;

    workersStarted = true;
    stopping = false;
    // Only the single submit thread is spawned here. The convert worker pool is
    // CPU-path-only and spawned lazily (ensureConvertPipeline) on the first
    // frame that actually needs a CPU swscale — a GPU/NVENC-only encoder never
    // spawns it and pays nothing for idle convert threads.
    encodeThread = std::thread([this] { encodeSubmitLoop(); });
}

void VideoEncoder::ensureConvertPipeline()
{
    if (convertStarted)
        return;

    // Pools sized exactly to the in-flight cap: the caller acquires a staging
    // buffer + a YUV frame under the same lock as the inFlight bump, so at most
    // maxFramesInFlight of each are ever checked out at once.
    const size_t poolSize = maxFramesInFlight;
    stagingPool.reserve(poolSize);
    yuvPool.reserve(poolSize);
    for (size_t i = 0; i < poolSize; ++i)
    {
        stagingPool.push_back(std::make_unique<std::vector<uint8_t>>());
        {
            std::lock_guard<std::mutex> lk(mu);
            freeStaging.push_back(stagingPool.back().get());
        }

        auto f = std::make_unique<nelux::Frame>();
        f->get()->format = outputPixelFormat;
        f->get()->width = width;
        f->get()->height = height;
        f->allocateBuffer(32);
        {
            std::lock_guard<std::mutex> lk(mu);
            freeYuv.push_back(f.get());
        }
        yuvPool.push_back(std::move(f));
    }

    // One converter (own SwsContext) per convert worker — SwsContext is not
    // safe to share across threads.
    converters.reserve(numConvertWorkers);
    for (int i = 0; i < numConvertWorkers; ++i)
        converters.push_back(std::make_unique<nelux::conversion::cpu::RGBToAutoConverter>(
            width, height, outputPixelFormat, props.colorspace, props.colorRange));

    for (int i = 0; i < numConvertWorkers; ++i)
        convertWorkers.emplace_back([this, i] { convertWorkerLoop(i); });

    convertStarted = true;
}

void VideoEncoder::convertWorkerLoop(int workerId)
{
    auto* conv = converters[workerId].get();
    for (;;)
    {
        ConvertJob job;
        {
            std::unique_lock<std::mutex> lk(mu);
            cvConvert.wait(lk, [&] { return !convertQueue.empty() || stopping; });
            if (convertQueue.empty())
            {
                if (stopping)
                    break;       // drained + stopping -> exit
                continue;
            }
            job = convertQueue.front();
            convertQueue.pop_front();
        }

        bool converted = true;
        try
        {
            // Each YUV frame is recycled; the encoder may still reference a prior
            // submission (B-frames), so make_writable does a COW realloc when needed.
            job.yuv->get()->pts = AV_NOPTS_VALUE;
            if (av_frame_make_writable(job.yuv->get()) < 0)
                throw std::runtime_error("Failed to make encoder frame writable");

            conv->convert(*job.yuv, job.staging->data());
        }
        catch (...)
        {
            converted = false;
            std::lock_guard<std::mutex> lk(mu);
            if (!workerError)
                workerError = std::current_exception();
            // Conversion failed: recycle the buffers and do NOT hand a partial
            // frame to the submit thread (it would encode garbage). The submit
            // thread observes workerError via cvReady and tears down.
            freeStaging.push_back(job.staging);
            freeYuv.push_back(job.yuv);
            --inFlight;
        }
        if (!converted)
        {
            cvReady.notify_one();
            cvFree.notify_all();
            continue;
        }

        {
            std::lock_guard<std::mutex> lk(mu);
            readyMap[job.seq].yuv = job.yuv;       // hand off to the submit thread
            freeStaging.push_back(job.staging);    // RGB buffer free immediately
        }
        cvReady.notify_one();
        cvFree.notify_one();
    }
}

void VideoEncoder::submitYuvFrame(nelux::Frame* yuv)
{
    encoder->encodeFrame(*yuv);
}

void VideoEncoder::encodeSubmitLoop()
{
    for (;;)
    {
        ReadyEntry entry;
        {
            std::unique_lock<std::mutex> lk(mu);
            cvReady.wait(lk, [&] {
                return readyMap.count(nextSubmitSeq) || workerError ||
                       (stopping && nextSubmitSeq >= enqueueSeq);
            });
            auto it = readyMap.find(nextSubmitSeq);
            if (it == readyMap.end())
            {
                // Nothing ready for the next slot. Exit only once stopping AND we
                // have submitted everything admitted; otherwise (e.g. error) bail.
                if (workerError || (stopping && nextSubmitSeq >= enqueueSeq))
                {
                    // Dispose anything still pending so CUDA events and GPU
                    // tensors don't leak when we exit early on error. (Holding mu.)
                    for (auto& kv : readyMap)
                    {
#ifdef NELUX_ENABLE_CUDA
                        if (kv.second.readyEvent)
                            cudaEventDestroy(kv.second.readyEvent);
                        if (kv.second.gpuTensor.defined())
                            retiredTensors.push_back(std::move(kv.second.gpuTensor));
#endif
                        if (kv.second.yuv)
                            freeYuv.push_back(kv.second.yuv);
                    }
                    readyMap.clear();
                    inFlight = 0;
                    break;
                }
                continue;
            }
            entry = std::move(it->second);
            readyMap.erase(it);
        }

        try
        {
#ifdef NELUX_ENABLE_CUDA
            if (entry.isGpu)
                submitGpuToEncoder(entry.gpuTensor, entry.readyEvent);
            else
#endif
                submitYuvFrame(entry.yuv);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lk(mu);
            if (!workerError)
                workerError = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lk(mu);
#ifdef NELUX_ENABLE_CUDA
            if (entry.gpuTensor.defined())
                retiredTensors.push_back(std::move(entry.gpuTensor));  // freed by caller under GIL
#endif
            if (entry.yuv)
                freeYuv.push_back(entry.yuv);
            ++nextSubmitSeq;
            --inFlight;
        }
        cvFree.notify_all();
    }
}

#ifdef NELUX_ENABLE_CUDA
void VideoEncoder::submitGpuToEncoder(torch::Tensor& gpuTensor, cudaEvent_t readyEvent)
{
    // Destroy the producer-ready event on every exit path (any call below can
    // throw). Declared first so it also covers a cudaSetDevice failure; the
    // parameter is nulled so nothing double-frees it.
    struct EventGuard
    {
        cudaEvent_t e;
        ~EventGuard() { if (e) cudaEventDestroy(e); }
    } eventGuard{readyEvent};
    readyEvent = nullptr;
    const cudaEvent_t producerReady = eventGuard.e;

    int deviceIndex = gpuTensor.device().index();
    if (deviceIndex < 0)
        deviceIndex = 0;
    if (cudaError_t cerr = cudaSetDevice(deviceIndex); cerr != cudaSuccess)
        throw std::runtime_error("Failed to select CUDA device for NVENC encode: " +
                                 std::string(cudaGetErrorString(cerr)));

    // Lazy one-time init of the encode stream + GPU converter, owned by the
    // worker thread that exclusively uses them.
    if (!encoderStream)
    {
        if (cudaStreamCreateWithFlags(&encoderStream, cudaStreamNonBlocking) != cudaSuccess)
            throw std::runtime_error("Failed to create NVENC CUDA stream");
    }
    if (!gpuConverter)
    {
        gpuConverter = std::make_unique<nelux::conversion::gpu::RGBToAutoGPUConverter>(
            width, height, outputPixelFormat, encoderStream);

        int gpuCs;
        switch (props.colorspace)
        {
        case AVCOL_SPC_BT709:
            gpuCs = nelux::backends::cuda::ColorSpaceEncode_BT709;
            break;
        case AVCOL_SPC_BT2020_NCL:
        case AVCOL_SPC_BT2020_CL:
            gpuCs = nelux::backends::cuda::ColorSpaceEncode_BT2020;
            break;
        case AVCOL_SPC_BT470BG:
        case AVCOL_SPC_SMPTE170M:
            gpuCs = nelux::backends::cuda::ColorSpaceEncode_BT601;
            break;
        default:
            gpuCs = nelux::backends::cuda::ColorSpaceEncode_BT709;
            break;
        }
        gpuConverter->setColorSpace(gpuCs);
        gpuConverter->setColorRange(props.colorRange == AVCOL_RANGE_JPEG
                                        ? nelux::backends::cuda::ColorRangeEncode_Full
                                        : nelux::backends::cuda::ColorRangeEncode_Limited);
    }

    // Make the encode stream wait until the producer (decode + dtype ops) has
    // finished writing the tensor, so the convert below never reads stale data.
    if (producerReady)
        cudaStreamWaitEvent(encoderStream, producerReady, 0);

    // RGB24 -> NV12/YUV on the GPU (writes into the converter's CUDA buffer).
    gpuConverter->convert(
        reinterpret_cast<const uint8_t*>(gpuTensor.data_ptr<uint8_t>()),
        width * 3);  // RGB24 pitch

    AVBufferRef* hwFramesCtx = encoder->getHwFramesCtx();
    if (!hwFramesCtx)
        throw std::runtime_error("NVENC hardware frames context not initialized");

    nelux::Frame hwFrame(hwFramesCtx);
    hwFrame.get()->format = AV_PIX_FMT_CUDA;
    hwFrame.get()->width = width;
    hwFrame.get()->height = height;

    // Device-to-device copy into the CUDA AVFrame, then sync THIS stream (the
    // worker blocks here, not the caller) before handing the frame to NVENC.
    gpuConverter->copyToCudaFrame(hwFrame.get());
    gpuConverter->synchronize();

    encoder->encodeFrame(hwFrame);  // eventGuard frees producerReady on return/throw
}
#endif

void VideoEncoder::stopEncodeWorkers()
{
    if (!workersStarted)
        return;
    {
        std::lock_guard<std::mutex> lk(mu);
        stopping = true;
    }
    cvConvert.notify_all();
    cvReady.notify_all();
    cvFree.notify_all();

    auto joinAll = [&] {
        // Join convert workers FIRST so they drain convertQueue and populate
        // readyMap for every admitted frame; then the submit thread can finish
        // sending all of them and exit.
        for (auto& t : convertWorkers)
            if (t.joinable())
                t.join();
        if (encodeThread.joinable())
            encodeThread.join();
    };

    // Release the GIL while joining: the submit thread may need it to free torch
    // tensors (GPU path); holding it here would deadlock the join.
    if (PyGILState_Check())
    {
        py::gil_scoped_release rel;
        joinAll();
    }
    else
    {
        joinAll();
    }

    convertWorkers.clear();
    workersStarted = false;
}

void VideoEncoder::close()
{
    // Drain + join the encode workers FIRST so every queued frame is sent to the
    // codec before we flush. Flushing (encoder->close sends a null frame) must
    // happen only after the last real frame.
    stopEncodeWorkers();

#ifdef NELUX_ENABLE_CUDA
    if (gpuConverter)
    {
        gpuConverter->synchronize();
        gpuConverter.reset();
    }

    if (encoderStream)
    {
        cudaStreamDestroy(encoderStream);
        encoderStream = nullptr;
    }
#endif

    if (encoder)
    {
        encoder->close();
        encoder.reset();
    }

#ifdef NELUX_ENABLE_CUDA
    // Drain any GPU tensors the submit thread retired but the caller didn't get
    // to free. close() is normally called from Python (GIL held); guard for the
    // destructor-at-shutdown path.
    if (!retiredTensors.empty())
    {
        if (PyGILState_Check())
        {
            retiredTensors.clear();
        }
        else
        {
            pybind11::gil_scoped_acquire gil;
            retiredTensors.clear();
        }
    }
#endif

    // Surface any error the worker hit mid-stream (it kept draining to avoid a
    // producer deadlock; this is the first place we can throw cleanly).
    if (workerError)
    {
        std::exception_ptr e = workerError;
        workerError = nullptr;
        std::rethrow_exception(e);
    }
}

VideoEncoder::~VideoEncoder()
{
    // close() can rethrow a worker error; a throwing destructor would terminate.
    // The worker is always joined here so no thread outlives the object.
    try
    {
        close();
    }
    catch (...)
    {
        // Best-effort: ensure the workers are joined even if flush threw.
        try { stopEncodeWorkers(); } catch (...) {}
    }
}

} // namespace nelux
