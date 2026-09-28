#pragma once

#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace nelux::tensor {
using Tensor = torch::stable::Tensor;
using Device = torch::stable::Device;
using Dtype = torch::headeronly::ScalarType;
using Shape = torch::headeronly::IntHeaderOnlyArrayRef;
inline constexpr auto kCPU = torch::headeronly::DeviceType::CPU;
inline constexpr auto kCUDA = torch::headeronly::DeviceType::CUDA;
inline constexpr auto kUInt8 = Dtype::Byte;
inline constexpr auto kUInt16 = Dtype::UInt16;
inline constexpr auto kUInt32 = Dtype::UInt32;
inline constexpr auto kInt8 = Dtype::Char;
inline constexpr auto kInt16 = Dtype::Short;
inline constexpr auto kInt32 = Dtype::Int;
inline constexpr auto kInt64 = Dtype::Long;
inline constexpr auto kFloat16 = Dtype::Half;
inline constexpr auto kFloat32 = Dtype::Float;
inline constexpr auto kFloat64 = Dtype::Double;

inline Tensor empty(Shape shape, Dtype dtype, Device device = Device(kCPU)) {
    return torch::stable::empty(shape, dtype, std::nullopt, device);
}
inline Tensor to(const Tensor& t, Dtype dtype) {
    return torch::stable::to(t, dtype);
}
inline Tensor to(const Tensor& t, Device device, bool non_blocking = false, bool copy = false) {
    return torch::stable::to(t, device, non_blocking, copy);
}
inline Tensor to(const Tensor& t, Device device, Dtype dtype) {
    return torch::stable::to(t, dtype, std::nullopt, device);
}
using torch::stable::clone;
using torch::stable::contiguous;
using torch::stable::empty_like;
using torch::stable::reshape;
using torch::stable::select;
using torch::stable::squeeze;
using torch::stable::unsqueeze;
using torch::stable::view;
inline Tensor narrow(Tensor t, int64_t dim, int64_t start, int64_t length) {
    return torch::stable::narrow(t, dim, start, length);
}
inline Tensor copy_(Tensor destination, const Tensor& source, bool non_blocking = false) {
    return torch::stable::copy_(destination, source, non_blocking);
}
inline bool is_floating_point(const Tensor& t) {
    const auto dtype = t.scalar_type();
    return dtype == Dtype::Half || dtype == Dtype::Float || dtype == Dtype::Double ||
        dtype == Dtype::BFloat16 || dtype == Dtype::Float8_e5m2 ||
        dtype == Dtype::Float8_e4m3fn || dtype == Dtype::Float8_e5m2fnuz ||
        dtype == Dtype::Float8_e4m3fnuz || dtype == Dtype::Float8_e8m0fnu ||
        dtype == Dtype::Float4_e2m1fn_x2;
}
template <class T> T* data_ptr(const Tensor& t) { return t.mutable_data_ptr<T>(); }

Tensor permute(const Tensor& t, Shape dims);
Tensor stack(const std::vector<Tensor>& tensors, int64_t dim = 0);
Tensor index_select(const Tensor& t, int64_t dim, const Tensor& indices);
Tensor arange(int64_t end, Dtype dtype = kInt64, Device device = Device(kCPU));
Tensor round(const Tensor& t);
Tensor clamp(const Tensor& t, int64_t minimum, int64_t maximum);
Tensor mul_integer(const Tensor& t, int64_t value);
Tensor mul_floating(const Tensor& t, double value);
Tensor div_integer(const Tensor& t, int64_t value);
Tensor div_floating(const Tensor& t, double value);
template<class S, std::enable_if_t<std::is_integral_v<S>, int> = 0>
Tensor mul(const Tensor& t, S value) { return mul_integer(t, static_cast<int64_t>(value)); }
template<class S, std::enable_if_t<std::is_floating_point_v<S>, int> = 0>
Tensor mul(const Tensor& t, S value) { return mul_floating(t, static_cast<double>(value)); }
template<class S, std::enable_if_t<std::is_integral_v<S>, int> = 0>
Tensor div(const Tensor& t, S value) { return div_integer(t, static_cast<int64_t>(value)); }
template<class S, std::enable_if_t<std::is_floating_point_v<S>, int> = 0>
Tensor div(const Tensor& t, S value) { return div_floating(t, static_cast<double>(value)); }

// The C shim takes ownership of its deleter context only once a Tensor exists.
// Track a synchronous callback on construction failure, so the captured native
// pool lease is released once whether validation fails or storage dies later.
template<class F>
Tensor from_blob(void* data, Shape shape, F deleter, Dtype dtype, Device device = Device(kCPU)) {
    struct Context {
        F deleter;
        std::atomic<bool> released{false};
        // Construction and the pending storage callback each own one reference.
        std::atomic<unsigned> references{2};
        explicit Context(const F& f) : deleter(f) {}
        void releaseReference() {
            if (references.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this;
        }
    };
    struct ReleaseReference {
        Context* context;
        ~ReleaseReference() { context->releaseReference(); }
    };
    Context* context = nullptr;
    bool committed = false;
    struct ReleaseOnFailure {
        F& deleter;
        void* data;
        Context*& context;
        bool& committed;
        ~ReleaseOnFailure() {
            if (!context) {
                deleter(data);
                return;
            }
            ReleaseReference construction{context};
            if ((!committed || !data) && !context->released.exchange(true, std::memory_order_acq_rel)) {
                ReleaseReference native{context};
                deleter(data);
            }
        }
    } release{deleter, data, context, committed};
    // Keep the original callable valid through every preliminary allocation.
    context = new Context(deleter);
    std::array<int64_t, 8> inline_strides;
    std::vector<int64_t> large_strides;
    int64_t* strides = inline_strides.data();
    if (shape.size() > inline_strides.size()) {
        large_strides.resize(shape.size());
        strides = large_strides.data();
    }
    int64_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        if (shape[i - 1] < 0 || (shape[i - 1] > 0 && stride > std::numeric_limits<int64_t>::max() / shape[i - 1])) {
            throw std::invalid_argument("Invalid shape for native frame tensor");
        }
        strides[i - 1] = stride;
        stride *= std::max<int64_t>(shape[i - 1], 1);
    }
    const auto shim_dtype = torch::stable::detail::to<int32_t>(torch::stable::detail::from(dtype));
    const auto shim_device = torch::stable::detail::to<int32_t>(torch::stable::detail::from(device.type()));
    const auto shim_layout = torch::stable::detail::to<int32_t>(
        torch::stable::detail::from(torch::headeronly::Layout::Strided));
    AtenTensorHandle handle = nullptr;
    const auto error = torch_from_blob(data, shape.size(), shape.data(), strides, 0,
        shim_dtype, shim_device,
        device.index(), &handle,
        shim_layout,
        nullptr, 0,
        [](void* pointer, void* opaque) {
            auto* owner = static_cast<Context*>(opaque);
            if (!owner->released.exchange(true, std::memory_order_acq_rel)) {
                ReleaseReference native{owner};
                owner->deleter(pointer);
            }
        }, context);
    TORCH_ERROR_CODE_CHECK(error);
    // shared_ptr construction inside Tensor(handle) deletes the native handle
    // if its control-block allocation fails, invoking the callback itself.
    committed = true;
    Tensor result(handle);
    // For null data the floor shim creates empty_strided storage and does not
    // install the callback. Own its native handle before invoking the original
    // lease, so a throwing deleter cannot leak that newly allocated storage.
    if (!data && !context->released.exchange(true, std::memory_order_acq_rel)) {
        ReleaseReference native{context};
        deleter(data);
    }
    return result;
}
} // namespace nelux::tensor
