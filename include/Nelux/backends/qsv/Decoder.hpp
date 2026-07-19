// QSV Decoder.hpp - Intel Quick Sync Video hardware-accelerated decoder
#pragma once

#include "backends/Decoder.hpp"

extern "C" {
#include <libavutil/hwcontext.h>
}

namespace nelux::backends::qsv
{

/**
 * @brief Intel Quick Sync Video (QSV) hardware-accelerated decoder.
 *
 * Uses FFmpeg's *_qsv decoders (h264_qsv, hevc_qsv, av1_qsv, ...) backed by an
 * AV_HWDEVICE_TYPE_QSV device (oneVPL; on Windows the child device is D3D11VA,
 * on Linux VAAPI). The bitstream is decoded on the Intel GPU; frames are
 * returned in system memory as NV12/P010 via the decoder's GPU-assisted
 * copyback (gpu_copy=on), so the entire software conversion pipeline of the
 * base Decoder (swscale RGB/gray convert, resize, 10-bit, sync/async paths)
 * is reused unchanged.
 *
 * Output tensors are CPU tensors; VideoReader can optionally move them to a
 * torch device (e.g. "xpu") after decode. There is no zero-copy D3D11->torch
 * path: on integrated GPUs the copyback crosses shared DRAM, so its cost is a
 * memcpy, not a PCIe transfer.
 *
 * Motion-vector export is not available (hardware decoders produce no MV
 * side-data); VideoReader rejects motion_vectors=True with this backend.
 */
class Decoder : public nelux::Decoder
{
  public:
    Decoder(const std::string& filePath, int numThreads, bool syncMode = false,
            bool grayscale = false, bool motionVectors = false);

    Decoder(const std::string& filePath, int numThreads, int resizeWidth,
            int resizeHeight, bool syncMode = false, bool grayscale = false,
            int resizeFilter = SWS_BILINEAR, bool motionVectors = false);

    ~Decoder() override;

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

  protected:
    // Opens the codec-matched *_qsv decoder with an AV_HWDEVICE_TYPE_QSV
    // hw_device_ctx and system-memory (NV12/P010) output. Called from the base
    // initialize() and reconfigure() paths.
    void initCodecContext() override;

  private:
    // Owned QSV hardware device context; codecCtx holds its own reference.
    AVBufferRef* hwDeviceCtx_ = nullptr;
};

} // namespace nelux::backends::qsv
