#pragma once

#include <cstddef>
#include <cstdint>

namespace nelux
{
namespace conversion
{
namespace cpu
{

/**
 * Packed 8-bit RGB/RGBA rows read straight from a strided [H, W, C] CPU
 * tensor, so encode_frame does not run torch CPU ops on the frame first.
 *
 * Every torch op on a CPU tensor wakes ATen's OpenMP pool, which then spins
 * through the gap to the next frame: a CHW view (encode_video's
 * dimension_order="CHW" permutes the batch) or a float frame cost 45-60 ms
 * of CPU per 1080p frame that way, for a few milliseconds of work.
 *
 * The bytes are exactly what encode_frame's torch normalisation gives, so the
 * encoded stream does not change:
 *   uint8          the samples themselves
 *   uint16         trunc(float(x) / 257.0f)
 *   float32        trunc(clamp(x * 255.0f, 0, 255)), NaN -> 0
 *   float16/bf16   widened to float32 (exact), then as float32
 *   float64        rounded to float32, then as float32
 * These were checked against torch 2.14 for every uint16, float16, bfloat16
 * and float32 bit pattern; tests/test_encode_input_layouts.py checks the
 * encoded streams.
 *
 * loadRowsU16 does the same for a 10-16-bit destination, which encode_frame
 * stages as RGB48LE/RGBA64LE:
 *   uint16         the samples themselves
 *   float32        clamp(round_half_even(x * 65535.0f), 0, 65535), NaN -> 0
 *   float16/bf16   widened to float32 (exact), then as float32
 *   float64        rounded to float32, then as float32
 * the bytes of (x.to(float32) * 65535).round().clamp(0, 65535).to(uint16),
 * checked against torch 2.12 and 2.14 for every float16, bfloat16 and
 * float32 bit pattern. uint8 is never staged 16-bit and is not read.
 */

enum class SampleType
{
    U8,
    U16,
    F16,
    BF16,
    F32,
    F64
};

/// Strides are in elements and must not be negative.
struct RowSource
{
    const uint8_t* data = nullptr;  // element [0, 0, 0]
    SampleType type = SampleType::U8;
    int channels = 3;               // 3 (RGB) or 4 (RGBA)
    int64_t rowStride = 0;
    int64_t pixelStride = 0;
    int64_t channelStride = 0;
};

/// True when loadRowsU8 and loadRowsU16 can read this layout: pixels packed within a row
/// (pixelStride == channels, channelStride == 1) or one plane per channel
/// (pixelStride == 1). Rows may lie anywhere.
bool rowLoaderCovers(const RowSource& src);

/// True when the rows are already packed 8-bit RGB/RGBA back to back, so a
/// consumer can read the tensor's memory as is.
bool isPackedU8(const RowSource& src, int width);

/// Rows [y0, y1) of @p src as packed 8-bit pixels (channels bytes each), row
/// y at dst + (y - y0) * dstStride. Uses AVX2 (and F16C for float16) when the
/// CPU has them; the bytes do not depend on it.
void loadRowsU8(const RowSource& src, int width, int y0, int y1, uint8_t* dst,
                ptrdiff_t dstStride);

/// Scalar model of loadRowsU8: what the vector code is tested against.
void loadRowsU8Reference(const RowSource& src, int width, int y0, int y1, uint8_t* dst,
                         ptrdiff_t dstStride);

/// True when the rows are already packed 16-bit RGB48/RGBA64 back to back.
bool isPackedU16(const RowSource& src, int width);

/// Rows [y0, y1) of @p src as packed little-endian 16-bit pixels (2 *
/// channels bytes each), row y at dst + (y - y0) * dstStride bytes. @p src
/// must not be uint8. Uses AVX2 (and F16C for float16) when the CPU has
/// them; the bytes do not depend on it.
void loadRowsU16(const RowSource& src, int width, int y0, int y1, uint8_t* dst,
                 ptrdiff_t dstStride);

/// Scalar model of loadRowsU16.
void loadRowsU16Reference(const RowSource& src, int width, int y0, int y1, uint8_t* dst,
                          ptrdiff_t dstStride);

} // namespace cpu
} // namespace conversion
} // namespace nelux
