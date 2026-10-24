#pragma once
#include <torch/torch.h>

namespace nelux {
// Baseline JPEG from a CPU uint8 HW or HWC image (1 or 3 channels) with
// libjpeg-turbo. subsampling is 444, 422 or 420 and is ignored for one channel.
// Larger images are encoded as independent horizontal strips in parallel and
// joined with restart markers. The strip layout depends only on the image
// geometry, so the bytes are the same at any thread count.
torch::Tensor encodeJpegCPU(torch::Tensor image, int quality, int subsampling);
}
