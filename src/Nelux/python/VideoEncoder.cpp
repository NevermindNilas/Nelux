#include "python/VideoEncoder.hpp"
#include <filesystem>
#include <stdexcept>
#include <Factory.hpp>
#include <cpu/RGBToAutoLibyuv.hpp>

namespace fs = std::filesystem;

namespace nelux
{
    //NOTE --- USED HWC
VideoEncoder::VideoEncoder(const std::string& filename,
                           std::optional<std::string> codec, std::optional<int> width,
                           std::optional<int> height, std::optional<int> bitRate,
                           std::optional<float> fps, std::optional<int> audioBitRate,
                           std::optional<int> audioSampleRate,
                           std::optional<int> audioChannels,
                           std::optional<std::string> audioCodec,
                           std::optional<int> preset,
                           std::optional<int> cq,
                           std::optional<std::string> pixelFormat)
{
    auto properties = inferEncodingProperties(filename, codec, width, height, bitRate,
                                              fps, audioBitRate, audioSampleRate,
                                              audioChannels, audioCodec,
                                              preset, cq, pixelFormat);
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
    std::optional<float> fps, std::optional<int> audioBitRate,
    std::optional<int> audioSampleRate, std::optional<int> audioChannels,
    std::optional<std::string> audioCodec,
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

    // Populate audio encoding settings (0 → no audio)
    if (audioBitRate.has_value() && audioSampleRate.has_value() &&
        audioChannels.has_value() && audioCodec.has_value())
    {
        props.audioBitRate = *audioBitRate;
        props.audioSampleRate = *audioSampleRate;
        props.audioChannels = *audioChannels;
        props.audioCodec = *audioCodec;
    }
    else
    {
        props.audioBitRate = 0;
        props.audioSampleRate = 0;
        props.audioChannels = 0;
        props.audioCodec = std::string();
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

    // Use libyuv converter for CPU path
    if (!converter)
    { 
        converter = std::make_unique<nelux::conversion::cpu::RGBToAutoLibyuvConverter>(
            width, height, outputPixelFormat);
    }

    // Convert RGB24 → YUV (I420 or NV12)
    converter->convert(convertedFrame, frame.data_ptr<uint8_t>());

    // Send converted AVFrame to encoder
    encoder->encodeFrame(convertedFrame);
}

void VideoEncoder::close()
{
    if (encoder)
    {
        encoder->close();
        encoder.reset();
    }
}

VideoEncoder::~VideoEncoder()
{
}

// video side stays the same…

/// Replace your old single‑frame binding
void VideoEncoder::encodeAudioFrame(const torch::Tensor& pcm)
{
    py::gil_scoped_release release;

    // 1) grab properties
    auto& props = encoder->Properties();
    int channels = props.audioChannels;
    int sampleRate = props.audioSampleRate;
    int frameSz = encoder->audioFrameSize();
    if (!frameSz)
        throw std::runtime_error("audioFrameSize not set");

    // 2) move to CPU, int16, contiguous
    // move to CPU and cast to Int16 in one go:
    auto t = pcm.to(torch::Device(torch::kCPU), // <— device
                    torch::kInt16,              // <— dtype
                    /*non_blocking=*/false,     // optional
                    /*copy=*/false)             // optional
                 .contiguous();

    auto ptr = t.data_ptr<int16_t>();
    int64_t totalSamples = t.numel() / channels;
    int64_t offset = 0;

    // 3) loop over full‑sized (and final smaller) frames
    while (offset < totalSamples)
    {
        int thisCount = std::min<int64_t>(frameSz, totalSamples - offset);

        // build an AVFrame
        nelux::Frame af;
        AVFrame* f = af.get();
        f->nb_samples = thisCount;
        f->sample_rate = sampleRate;
        f->format = AV_SAMPLE_FMT_FLTP; // planar float for AAC
        av_channel_layout_default(&f->ch_layout, channels);
        af.allocateBuffer(0);

        // de‑interleave & convert
        std::vector<float> buf(channels * thisCount);
        for (int ch = 0; ch < channels; ++ch)
        {
            float* dst = buf.data() + ch * thisCount;
            int16_t* src = ptr + (offset * channels) + ch;
            for (int i = 0; i < thisCount; ++i)
                dst[i] = src[i * channels] / 32768.0f;
            std::memcpy(f->data[ch], dst, thisCount * sizeof(float));
        }

        // call the low‑level API
        if (!encoder->encodeAudioFrame(af))
            throw std::runtime_error("audio encode failed");

        offset += thisCount;
    }
}


} // namespace nelux
