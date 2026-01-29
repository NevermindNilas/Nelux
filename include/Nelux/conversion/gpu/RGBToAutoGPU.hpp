/*
 * RGBToAutoGPU.hpp - GPU-based RGB to YUV conversion for encoding
 * 
 * This converter allocates CUDA buffers for NV12/YUV output and runs
 * the RGB→YUV kernel entirely on GPU. The resulting CUDA buffer can
 * then be uploaded to NVENC via av_hwframe_transfer_data.
 * 
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifdef NELUX_ENABLE_CUDA

#include <Nelux/backends/cuda/RGBToNV12.cuh>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixfmt.h>
}

namespace nelux::conversion::gpu
{

/**
 * @brief GPU-based RGB24 to YUV converter for encoding
 * 
 * Allocates CUDA buffers for the output YUV format. After conversion,
 * the data needs to be copied to AVFrame via getCudaBuffer() + memcpy
 * or av_hwframe_transfer_data.
 */
class RGBToAutoGPUConverter
{
private:
    int width;
    int height;
    AVPixelFormat dst_fmt;
    cudaStream_t stream;
    
    // Color space settings (default to BT.709 limited for HD content)
    int colorSpace = nelux::backends::cuda::ColorSpaceEncode_BT709;
    int colorRange = nelux::backends::cuda::ColorRangeEncode_Limited;
    
    // CUDA buffers for NV12/YUV output
    uint8_t* cudaNv12Buffer = nullptr;  // For NV12/P010/NV16
    uint8_t* cudaYBuffer = nullptr;     // For YUV444P (Y plane)
    uint8_t* cudaUBuffer = nullptr;     // For YUV444P (U plane)
    uint8_t* cudaVBuffer = nullptr;     // For YUV444P (V plane)
    
    size_t bufferSize = 0;
    int nv12Pitch = 0;
    int surfaceHeight = 0;

public:
    RGBToAutoGPUConverter(int w, int h, AVPixelFormat format, cudaStream_t cudaStream = nullptr)
        : width(w), height(h), dst_fmt(format), stream(cudaStream)
    {
        // Validate supported formats and allocate buffers
        switch (format) {
            case AV_PIX_FMT_NV12:
                allocateNv12Buffer();
                break;
            case AV_PIX_FMT_P010LE:
                allocateP010Buffer();
                break;
            case AV_PIX_FMT_YUV444P:
                allocateYuv444Buffer();
                break;
            case AV_PIX_FMT_NV16:
                allocateNv16Buffer();
                break;
            default:
                throw std::runtime_error("RGBToAutoGPUConverter: Unsupported pixel format");
        }
    }
    
    ~RGBToAutoGPUConverter()
    {
        freeBuffers();
    }
    
    // No copy
    RGBToAutoGPUConverter(const RGBToAutoGPUConverter&) = delete;
    RGBToAutoGPUConverter& operator=(const RGBToAutoGPUConverter&) = delete;
    
    /**
     * @brief Set color space for conversion
     */
    void setColorSpace(int space) {
        colorSpace = space;
    }
    
    /**
     * @brief Set color range for conversion
     */
    void setColorRange(int range) {
        colorRange = range;
    }
    
    /**
     * @brief Convert GPU RGB24 buffer to YUV in CUDA buffer
     * 
     * After calling this, use copyToCpuFrame() to get data into AVFrame.
     * 
     * @param gpuRgb Pointer to RGB24 data in GPU memory (HWC format)
     * @param rgbPitch Pitch of RGB buffer in bytes (typically width * 3)
     */
    void convert(const uint8_t* gpuRgb, int rgbPitch = 0)
    {
        if (!gpuRgb) {
            throw std::runtime_error("RGBToAutoGPUConverter::convert: null pointer");
        }
        
        if (rgbPitch == 0) {
            rgbPitch = width * 3;
        }
        
        switch (dst_fmt) {
            case AV_PIX_FMT_NV12:
                nelux::backends::cuda::Rgb24ToNv12(
                    gpuRgb, rgbPitch,
                    cudaNv12Buffer, nv12Pitch,
                    width, height, surfaceHeight,
                    colorSpace, colorRange, stream);
                break;
                
            case AV_PIX_FMT_P010LE:
                nelux::backends::cuda::Rgb24ToP010(
                    gpuRgb, rgbPitch,
                    cudaNv12Buffer, nv12Pitch,
                    width, height, surfaceHeight,
                    colorSpace, colorRange, stream);
                break;
                
            case AV_PIX_FMT_YUV444P:
                nelux::backends::cuda::Rgb24ToYuv444(
                    gpuRgb, rgbPitch,
                    cudaYBuffer, cudaUBuffer, cudaVBuffer,
                    width, width, height,
                    colorSpace, colorRange, stream);
                break;
                
            case AV_PIX_FMT_NV16:
                nelux::backends::cuda::Rgb24ToNv16(
                    gpuRgb, rgbPitch,
                    cudaNv12Buffer, nv12Pitch,
                    width, height, surfaceHeight,
                    colorSpace, colorRange, stream);
                break;
                
            default:
                throw std::runtime_error("RGBToAutoGPUConverter: Unsupported format");
        }
        
        // Synchronize to ensure conversion is complete
        if (stream) {
            cudaStreamSynchronize(stream);
        } else {
            cudaDeviceSynchronize();
        }
    }
    
    /**
     * @brief Copy converted YUV data from CUDA buffer to CPU AVFrame
     * 
     * This downloads the data from GPU to CPU for use with software
     * encoders or av_hwframe_transfer_data.
     */
    void copyToCpuFrame(AVFrame* frame)
    {
        if (!frame) {
            throw std::runtime_error("RGBToAutoGPUConverter::copyToCpuFrame: null frame");
        }
        
        switch (dst_fmt) {
            case AV_PIX_FMT_NV12:
                copyNv12ToCpuFrame(frame);
                break;
            case AV_PIX_FMT_P010LE:
                copyP010ToCpuFrame(frame);
                break;
            case AV_PIX_FMT_YUV444P:
                copyYuv444ToCpuFrame(frame);
                break;
            case AV_PIX_FMT_NV16:
                copyNv16ToCpuFrame(frame);
                break;
            default:
                break;
        }
    }

    /**
     * @brief Copy converted YUV data from CUDA buffer to a CUDA AVFrame
     *
     * This performs device-to-device copies into an AV_PIX_FMT_CUDA frame
     * allocated from a hw_frames_ctx (NVENC path). No PCIe transfer occurs.
     */
    void copyToCudaFrame(AVFrame* frame)
    {
        if (!frame) {
            throw std::runtime_error("RGBToAutoGPUConverter::copyToCudaFrame: null frame");
        }

        switch (dst_fmt) {
            case AV_PIX_FMT_NV12:
                copyNv12ToCudaFrame(frame);
                break;
            case AV_PIX_FMT_P010LE:
                copyP010ToCudaFrame(frame);
                break;
            case AV_PIX_FMT_YUV444P:
                copyYuv444ToCudaFrame(frame);
                break;
            case AV_PIX_FMT_NV16:
                copyNv16ToCudaFrame(frame);
                break;
            default:
                break;
        }
    }
    
    /**
     * @brief Get pointer to CUDA NV12 buffer (for direct NVENC upload)
     */
    uint8_t* getCudaBuffer() const { return cudaNv12Buffer; }
    
    int getPitch() const { return nv12Pitch; }
    int getSurfaceHeight() const { return surfaceHeight; }
    
private:
    void allocateNv12Buffer()
    {
        // NV12: Y plane (width * height) + UV plane (width * height/2)
        nv12Pitch = width;
        surfaceHeight = height;
        bufferSize = static_cast<size_t>(nv12Pitch) * height * 3 / 2;
        
        cudaError_t err = cudaMalloc(&cudaNv12Buffer, bufferSize);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA NV12 buffer: " + 
                                     std::string(cudaGetErrorString(err)));
        }
    }
    
    void allocateP010Buffer()
    {
        // P010: 16-bit per component
        nv12Pitch = width * 2;  // 2 bytes per Y sample
        surfaceHeight = height;
        bufferSize = static_cast<size_t>(nv12Pitch) * height * 3 / 2;
        
        cudaError_t err = cudaMalloc(&cudaNv12Buffer, bufferSize);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA P010 buffer");
        }
    }
    
    void allocateYuv444Buffer()
    {
        size_t planeSize = static_cast<size_t>(width) * height;
        
        cudaError_t err = cudaMalloc(&cudaYBuffer, planeSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate Y buffer");
        
        err = cudaMalloc(&cudaUBuffer, planeSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate U buffer");
        
        err = cudaMalloc(&cudaVBuffer, planeSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate V buffer");
    }
    
    void allocateNv16Buffer()
    {
        // NV16: Y plane (width * height) + UV plane (width * height)
        nv12Pitch = width;
        surfaceHeight = height;
        bufferSize = static_cast<size_t>(nv12Pitch) * height * 2;  // 4:2:2 = 2x luma size
        
        cudaError_t err = cudaMalloc(&cudaNv12Buffer, bufferSize);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA NV16 buffer");
        }
    }
    
    void freeBuffers()
    {
        if (cudaNv12Buffer) { cudaFree(cudaNv12Buffer); cudaNv12Buffer = nullptr; }
        if (cudaYBuffer) { cudaFree(cudaYBuffer); cudaYBuffer = nullptr; }
        if (cudaUBuffer) { cudaFree(cudaUBuffer); cudaUBuffer = nullptr; }
        if (cudaVBuffer) { cudaFree(cudaVBuffer); cudaVBuffer = nullptr; }
    }
    
    void copyNv12ToCpuFrame(AVFrame* frame)
    {
        // Copy Y plane
        cudaMemcpy2D(
            frame->data[0], frame->linesize[0],
            cudaNv12Buffer, nv12Pitch,
            width, height,
            cudaMemcpyDeviceToHost);
        
        // Copy UV plane
        cudaMemcpy2D(
            frame->data[1], frame->linesize[1],
            cudaNv12Buffer + nv12Pitch * surfaceHeight, nv12Pitch,
            width, height / 2,
            cudaMemcpyDeviceToHost);
    }

    void copyNv12ToCudaFrame(AVFrame* frame)
    {
        cudaMemcpy2D(
            frame->data[0], frame->linesize[0],
            cudaNv12Buffer, nv12Pitch,
            width, height,
            cudaMemcpyDeviceToDevice);

        cudaMemcpy2D(
            frame->data[1], frame->linesize[1],
            cudaNv12Buffer + nv12Pitch * surfaceHeight, nv12Pitch,
            width, height / 2,
            cudaMemcpyDeviceToDevice);
    }
    
    void copyP010ToCpuFrame(AVFrame* frame)
    {
        // Copy Y plane (16-bit)
        cudaMemcpy2D(
            frame->data[0], frame->linesize[0],
            cudaNv12Buffer, nv12Pitch,
            width * 2, height,
            cudaMemcpyDeviceToHost);
        
        // Copy UV plane
        cudaMemcpy2D(
            frame->data[1], frame->linesize[1],
            cudaNv12Buffer + nv12Pitch * surfaceHeight, nv12Pitch,
            width * 2, height / 2,
            cudaMemcpyDeviceToHost);
    }

    void copyP010ToCudaFrame(AVFrame* frame)
    {
        cudaMemcpy2D(
            frame->data[0], frame->linesize[0],
            cudaNv12Buffer, nv12Pitch,
            width * 2, height,
            cudaMemcpyDeviceToDevice);

        cudaMemcpy2D(
            frame->data[1], frame->linesize[1],
            cudaNv12Buffer + nv12Pitch * surfaceHeight, nv12Pitch,
            width * 2, height / 2,
            cudaMemcpyDeviceToDevice);
    }

    void copyYuv444ToCudaFrame(AVFrame* frame)
    {
        cudaMemcpy2D(
            frame->data[0], frame->linesize[0],
            cudaYBuffer, width,
            width, height,
            cudaMemcpyDeviceToDevice);

        cudaMemcpy2D(
            frame->data[1], frame->linesize[1],
            cudaUBuffer, width,
            width, height,
            cudaMemcpyDeviceToDevice);

        cudaMemcpy2D(
            frame->data[2], frame->linesize[2],
            cudaVBuffer, width,
            width, height,
            cudaMemcpyDeviceToDevice);
    }

    void copyNv16ToCudaFrame(AVFrame* frame)
    {
        cudaMemcpy2D(
            frame->data[0], frame->linesize[0],
            cudaNv12Buffer, nv12Pitch,
            width, height,
            cudaMemcpyDeviceToDevice);

        cudaMemcpy2D(
            frame->data[1], frame->linesize[1],
            cudaNv12Buffer + nv12Pitch * surfaceHeight, nv12Pitch,
            width, height,
            cudaMemcpyDeviceToDevice);
    }
    
    void copyYuv444ToCpuFrame(AVFrame* frame)
    {
        size_t planeSize = static_cast<size_t>(width) * height;
        cudaMemcpy(frame->data[0], cudaYBuffer, planeSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(frame->data[1], cudaUBuffer, planeSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(frame->data[2], cudaVBuffer, planeSize, cudaMemcpyDeviceToHost);
    }
    
    void copyNv16ToCpuFrame(AVFrame* frame)
    {
        // Copy Y plane
        cudaMemcpy2D(
            frame->data[0], frame->linesize[0],
            cudaNv12Buffer, nv12Pitch,
            width, height,
            cudaMemcpyDeviceToHost);
        
        // Copy UV plane (same height as Y for 4:2:2)
        cudaMemcpy2D(
            frame->data[1], frame->linesize[1],
            cudaNv12Buffer + nv12Pitch * surfaceHeight, nv12Pitch,
            width, height,
            cudaMemcpyDeviceToHost);
    }
};

} // namespace nelux::conversion::gpu

#endif // NELUX_ENABLE_CUDA
