#include "python/VideoEncoder.hpp"
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <Factory.hpp>
#include <cpu/RGBToAuto.hpp>
#include <cpu/RGBToAutoLibyuv.hpp>

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
                           std::optional<std::string> pixelFormat)
{
    auto properties = inferEncodingProperties(filename, codec, width, height, bitRate,
                                              fps, preset, cq, pixelFormat);
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
    std::optional<std::string> pixelFormat)
{
    // Populate video encoding settings
    nelux::Encoder::EncodingProperties props;
    props.codec = codec.value_or("h264_mf");
    props.width = width.value_or(1920);
    props.height = height.value_or(1080);
    props.bitRate = bitRate.value_or(4000000); // 4 Mbps default
    props.fps = static_cast<int>(std::round(fps.value_or(30.0f)));
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

    // libyuv provides fast paths only for the common 8-bit YUV planar/biplanar
    // formats; anything else (gray*, 10/12-bit YUV, packed RGB, GBRP, ProRes
    // targets like yuv422p10le) goes through swscale which supports the full
    // pix_fmt matrix.
    if (!converter)
    {
        bool libyuvSupported = false;
        switch (outputPixelFormat)
        {
            case AV_PIX_FMT_YUV420P:
            case AV_PIX_FMT_YUVJ420P:
            case AV_PIX_FMT_NV12:
            case AV_PIX_FMT_YUV422P:
            case AV_PIX_FMT_YUVJ422P:
            case AV_PIX_FMT_YUV444P:
            case AV_PIX_FMT_YUVJ444P:
                libyuvSupported = true;
                break;
            default:
                libyuvSupported = false;
                break;
        }
        if (libyuvSupported)
        {
            converter = std::make_unique<nelux::conversion::cpu::RGBToAutoLibyuvConverter>(
                width, height, outputPixelFormat);
        }
        else
        {
            converter = std::make_unique<nelux::conversion::cpu::RGBToAutoConverter>(
                width, height, outputPixelFormat);
        }
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
