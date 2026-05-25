#include "python/VideoEncoder.hpp"
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <Factory.hpp>
#include <cpu/RGBToAuto.hpp>

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

void VideoEncoder::encodeFrame(torch::Tensor frame)
{
    if (!encoder)
        throw std::runtime_error("Encoder is not initialized");

    py::gil_scoped_release release;
    
#ifdef NELUX_ENABLE_CUDA
    // GPU path: When tensor is on CUDA and we're using NVENC
    if (frame.device().is_cuda() && encoder->isHardwareEncoder())
    {
        int deviceIndex = frame.device().index();
        if (deviceIndex < 0)
        {
            deviceIndex = 0;
        }

        cudaError_t deviceErr = cudaSetDevice(deviceIndex);
        if (deviceErr != cudaSuccess)
        {
            throw std::runtime_error("Failed to select CUDA device for NVENC encode: " +
                                     std::string(cudaGetErrorString(deviceErr)));
        }

        if (!encoderStream)
        {
            cudaError_t streamErr = cudaStreamCreateWithFlags(&encoderStream,
                                                              cudaStreamNonBlocking);
            if (streamErr != cudaSuccess)
            {
                throw std::runtime_error("Failed to create NVENC CUDA stream: " +
                                         std::string(cudaGetErrorString(streamErr)));
            }
        }

        // Convert tensor dtype to uint8 if needed (on GPU)
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
        {
            frame = frame.contiguous();
        }
        
        // Create GPU converter if not exists
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
        
        // Convert RGB to NV12/YUV on GPU (writes to CUDA buffer)
        gpuConverter->convert(
            reinterpret_cast<const uint8_t*>(frame.data_ptr<uint8_t>()),
            width * 3);  // RGB24 pitch

        // Allocate a CUDA AVFrame from NVENC's hw_frames_ctx (zero-copy path)
        AVBufferRef* hwFramesCtx = encoder->getHwFramesCtx();
        if (!hwFramesCtx)
        {
            throw std::runtime_error("NVENC hardware frames context not initialized");
        }

        nelux::Frame hwFrame(hwFramesCtx);
        hwFrame.get()->format = AV_PIX_FMT_CUDA;
        hwFrame.get()->width = width;
        hwFrame.get()->height = height;

        // Copy from converter's CUDA buffer into the CUDA AVFrame (device-to-device)
        gpuConverter->copyToCudaFrame(hwFrame.get());
        gpuConverter->synchronize();

        // Send CUDA frame directly to encoder (no CPU upload)
        encoder->encodeFrame(hwFrame);
        return;
    }
#endif
    
    // CPU path (fallback for non-CUDA tensors or software encoders)
    
    // Ensure CPU frame is allocated
    if (!cpuFrame.get()->data[0])
    {
        cpuFrame.get()->format = outputPixelFormat;
        cpuFrame.get()->width = width;
        cpuFrame.get()->height = height;
        cpuFrame.allocateBuffer(32);
    }
    
    // Reset PTS because we reuse the frame (MUST happen every time)
    cpuFrame.get()->pts = AV_NOPTS_VALUE;
    
    nelux::Frame& convertedFrame = cpuFrame;

    // Move tensor to CPU if on CUDA
    if (frame.device().is_cuda())
    {
        frame = frame.to(torch::kCPU);
    }

    // Convert tensor dtype to uint8 if needed
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

    // All RGB→YUV conversion goes through libswscale. Covers the full pix_fmt
    // matrix (gray*, 10/12-bit YUV, packed RGB, GBRP, planar 10le, ProRes
    // targets) and honors props.colorspace / props.colorRange directly.
    if (!converter)
    {
        converter = std::make_unique<nelux::conversion::cpu::RGBToAutoConverter>(
            width, height, outputPixelFormat, props.colorspace, props.colorRange);
    }

    // cpuFrame is reused across calls. A software encoder with B-frames /
    // multithreading keeps a reference to a previously submitted frame, so
    // overwriting the buffer in place would corrupt an in-flight frame.
    // av_frame_make_writable does a copy-on-write reallocation only when the
    // buffer is still referenced.
    if (av_frame_make_writable(convertedFrame.get()) < 0)
    {
        throw std::runtime_error("Failed to make encoder frame writable");
    }

    // Convert RGB24 → YUV (I420 or NV12)
    converter->convert(convertedFrame, frame.data_ptr<uint8_t>());

    // Send converted AVFrame to encoder
    encoder->encodeFrame(convertedFrame);
}

void VideoEncoder::close()
{
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

    converter.reset();
}

VideoEncoder::~VideoEncoder()
{
    close();
}

} // namespace nelux
