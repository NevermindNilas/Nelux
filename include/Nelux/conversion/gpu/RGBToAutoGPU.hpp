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
    
    // CUDA buffers for various output formats
    uint8_t* cudaNv12Buffer = nullptr;  // For NV12/P010/NV16/P210/P216
    uint8_t* cudaYBuffer = nullptr;     // For planar YUV (Y plane)
    uint8_t* cudaUBuffer = nullptr;     // For planar YUV (U plane)
    uint8_t* cudaVBuffer = nullptr;     // For planar YUV (V plane)
    uint8_t* cudaRgbBuffer = nullptr;   // For packed RGB/BGR (BGR0, RGBA, etc.)
    
    size_t bufferSize = 0;
    int nv12Pitch = 0;
    int surfaceHeight = 0;
    int rgbPitch = 0;  // For RGB packed formats

public:
    RGBToAutoGPUConverter(int w, int h, AVPixelFormat format, cudaStream_t cudaStream = nullptr)
        : width(w), height(h), dst_fmt(format), stream(cudaStream)
    {
        // Validate supported formats and allocate buffers
        switch (format) {
            // 4:2:0 formats
            case AV_PIX_FMT_NV12:
                allocateNv12Buffer();
                break;
            case AV_PIX_FMT_P010LE:
                allocateP010Buffer();
                break;
            case AV_PIX_FMT_YUV420P:
                allocateYuv420pBuffer();
                break;
            
            // 4:2:2 formats
            case AV_PIX_FMT_NV16:
                allocateNv16Buffer();
                break;
            case AV_PIX_FMT_P210LE:
                allocateP210Buffer();
                break;
            case AV_PIX_FMT_P216LE:
                allocateP216Buffer();
                break;
            
            // 4:4:4 formats
            case AV_PIX_FMT_YUV444P:
                allocateYuv444Buffer();
                break;
            case AV_PIX_FMT_YUV444P10LE:
                allocateYuv444P10Buffer();
                break;
            case AV_PIX_FMT_YUV444P16LE:
                allocateYuv444P16Buffer();
                break;
            
            // Packed RGB/BGR formats (no conversion needed, just channel swap)
            case AV_PIX_FMT_BGR0:
            case AV_PIX_FMT_BGRA:
            case AV_PIX_FMT_RGB0:
            case AV_PIX_FMT_RGBA:
                allocatePackedRgbBuffer(format);
                break;
            
            // Planar RGB formats (GBR)
            case AV_PIX_FMT_GBRP:
                allocateGbrpBuffer();
                break;
            case AV_PIX_FMT_GBRP16LE:
                allocateGbrp16Buffer();
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
                
            case AV_PIX_FMT_YUV420P:
                nelux::backends::cuda::Rgb24ToYuv420p(
                    gpuRgb, rgbPitch,
                    cudaYBuffer, cudaUBuffer, cudaVBuffer,
                    width,  // Y pitch = width for 8-bit
                    width, height,
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
            case AV_PIX_FMT_YUV420P:
                copyYuv420pToCpuFrame(frame);
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
            case AV_PIX_FMT_YUV420P:
                copyYuv420pToCudaFrame(frame);
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

    void synchronize() const
    {
        cudaError_t err = stream ? cudaStreamSynchronize(stream) : cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("RGBToAutoGPUConverter: CUDA sync failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }
    
private:
    void copyPlane2DToCuda(uint8_t* dst, int dstPitch, const uint8_t* src, int srcPitch,
                           size_t widthBytes, size_t copyHeight)
    {
        cudaError_t err = stream
                              ? cudaMemcpy2DAsync(dst, dstPitch, src, srcPitch,
                                                  widthBytes, copyHeight,
                                                  cudaMemcpyDeviceToDevice, stream)
                              : cudaMemcpy2D(dst, dstPitch, src, srcPitch,
                                             widthBytes, copyHeight,
                                             cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("RGBToAutoGPUConverter: CUDA D2D copy failed: " +
                                     std::string(cudaGetErrorString(err)));
        }
    }

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
    
    void allocateYuv420pBuffer()
    {
        // YUV420P: Y plane (width * height) + U plane (width/2 * height/2) + V plane (width/2 * height/2)
        size_t ySize = static_cast<size_t>(width) * height;
        size_t uvSize = ySize / 4;  // Quarter size for U and V
        
        cudaError_t err = cudaMalloc(&cudaYBuffer, ySize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate Y buffer");
        
        err = cudaMalloc(&cudaUBuffer, uvSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate U buffer");
        
        err = cudaMalloc(&cudaVBuffer, uvSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate V buffer");
    }
    
    void allocateP210Buffer()
    {
        // P210: 16-bit per component, 4:2:2 (UV same height as Y)
        nv12Pitch = width * 2;  // 2 bytes per Y sample
        surfaceHeight = height;
        bufferSize = static_cast<size_t>(nv12Pitch) * height * 2;  // 4:2:2 = 2x luma
        
        cudaError_t err = cudaMalloc(&cudaNv12Buffer, bufferSize);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA P210 buffer");
        }
    }
    
    void allocateP216Buffer()
    {
        // P216: 16-bit per component, 4:2:2
        nv12Pitch = width * 2;
        surfaceHeight = height;
        bufferSize = static_cast<size_t>(nv12Pitch) * height * 2;
        
        cudaError_t err = cudaMalloc(&cudaNv12Buffer, bufferSize);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA P216 buffer");
        }
    }
    
    void allocateYuv444P10Buffer()
    {
        // YUV444P10: 16-bit planes with 10-bit data
        size_t planeSize = static_cast<size_t>(width) * height * 2;  // 2 bytes per sample
        
        cudaError_t err = cudaMalloc(&cudaYBuffer, planeSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate Y buffer");
        
        err = cudaMalloc(&cudaUBuffer, planeSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate U buffer");
        
        err = cudaMalloc(&cudaVBuffer, planeSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate V buffer");
    }
    
    void allocateYuv444P16Buffer()
    {
        // YUV444P16: 16-bit planes
        size_t planeSize = static_cast<size_t>(width) * height * 2;
        
        cudaError_t err = cudaMalloc(&cudaYBuffer, planeSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate Y buffer");
        
        err = cudaMalloc(&cudaUBuffer, planeSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate U buffer");
        
        err = cudaMalloc(&cudaVBuffer, planeSize);
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate V buffer");
    }
    
    void allocatePackedRgbBuffer(AVPixelFormat format)
    {
        // BGR0, BGRA, RGB0, RGBA - packed formats
        int bytesPerPixel = (format == AV_PIX_FMT_BGRA || format == AV_PIX_FMT_RGBA) ? 4 : 3;
        rgbPitch = width * bytesPerPixel;
        bufferSize = static_cast<size_t>(rgbPitch) * height;
        
        cudaError_t err = cudaMalloc(&cudaRgbBuffer, bufferSize);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate CUDA RGB packed buffer");
        }
    }
    
    void allocateGbrpBuffer()
    {
        // GBRP: 3 separate 8-bit planes (G, B, R order for FFmpeg)
        size_t planeSize = static_cast<size_t>(width) * height;
        
        cudaError_t err = cudaMalloc(&cudaYBuffer, planeSize);  // G
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate G buffer");
        
        err = cudaMalloc(&cudaUBuffer, planeSize);  // B
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate B buffer");
        
        err = cudaMalloc(&cudaVBuffer, planeSize);  // R
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate R buffer");
    }
    
    void allocateGbrp16Buffer()
    {
        // GBRP16: 3 separate 16-bit planes
        size_t planeSize = static_cast<size_t>(width) * height * 2;
        
        cudaError_t err = cudaMalloc(&cudaYBuffer, planeSize);  // G
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate G16 buffer");
        
        err = cudaMalloc(&cudaUBuffer, planeSize);  // B
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate B16 buffer");
        
        err = cudaMalloc(&cudaVBuffer, planeSize);  // R
        if (err != cudaSuccess) throw std::runtime_error("Failed to allocate R16 buffer");
    }
    
    void freeBuffers()
    {
        if (cudaNv12Buffer) { cudaFree(cudaNv12Buffer); cudaNv12Buffer = nullptr; }
        if (cudaYBuffer) { cudaFree(cudaYBuffer); cudaYBuffer = nullptr; }
        if (cudaUBuffer) { cudaFree(cudaUBuffer); cudaUBuffer = nullptr; }
        if (cudaVBuffer) { cudaFree(cudaVBuffer); cudaVBuffer = nullptr; }
        if (cudaRgbBuffer) { cudaFree(cudaRgbBuffer); cudaRgbBuffer = nullptr; }
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
        copyPlane2DToCuda(frame->data[0], frame->linesize[0], cudaNv12Buffer,
                          nv12Pitch, width, height);

        copyPlane2DToCuda(frame->data[1], frame->linesize[1],
                          cudaNv12Buffer + nv12Pitch * surfaceHeight, nv12Pitch,
                          width, height / 2);
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
        copyPlane2DToCuda(frame->data[0], frame->linesize[0], cudaNv12Buffer,
                          nv12Pitch, width * 2, height);

        copyPlane2DToCuda(frame->data[1], frame->linesize[1],
                          cudaNv12Buffer + nv12Pitch * surfaceHeight, nv12Pitch,
                          width * 2, height / 2);
    }

    void copyYuv444ToCudaFrame(AVFrame* frame)
    {
        copyPlane2DToCuda(frame->data[0], frame->linesize[0], cudaYBuffer, width,
                          width, height);

        copyPlane2DToCuda(frame->data[1], frame->linesize[1], cudaUBuffer, width,
                          width, height);

        copyPlane2DToCuda(frame->data[2], frame->linesize[2], cudaVBuffer, width,
                          width, height);
    }

    void copyNv16ToCudaFrame(AVFrame* frame)
    {
        copyPlane2DToCuda(frame->data[0], frame->linesize[0], cudaNv12Buffer,
                          nv12Pitch, width, height);

        copyPlane2DToCuda(frame->data[1], frame->linesize[1],
                          cudaNv12Buffer + nv12Pitch * surfaceHeight, nv12Pitch,
                          width, height);
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
    
    void copyYuv420pToCpuFrame(AVFrame* frame)
    {
        // Copy Y plane (full resolution)
        cudaMemcpy2D(
            frame->data[0], frame->linesize[0],
            cudaYBuffer, width,
            width, height,
            cudaMemcpyDeviceToHost);
        
        // Copy U plane (quarter resolution)
        cudaMemcpy2D(
            frame->data[1], frame->linesize[1],
            cudaUBuffer, width / 2,
            width / 2, height / 2,
            cudaMemcpyDeviceToHost);
        
        // Copy V plane (quarter resolution)
        cudaMemcpy2D(
            frame->data[2], frame->linesize[2],
            cudaVBuffer, width / 2,
            width / 2, height / 2,
            cudaMemcpyDeviceToHost);
    }
    
    void copyYuv420pToCudaFrame(AVFrame* frame)
    {
        copyPlane2DToCuda(frame->data[0], frame->linesize[0], cudaYBuffer, width,
                          width, height);

        copyPlane2DToCuda(frame->data[1], frame->linesize[1], cudaUBuffer,
                          width / 2, width / 2, height / 2);

        copyPlane2DToCuda(frame->data[2], frame->linesize[2], cudaVBuffer,
                          width / 2, width / 2, height / 2);
    }
};

} // namespace nelux::conversion::gpu

#endif // NELUX_ENABLE_CUDA
