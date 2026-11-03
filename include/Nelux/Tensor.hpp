#pragma once
// Nelux's tensor type, built on the PyTorch stable ABI.
//
// Nelux compiles against the stable C ABI (torch/csrc/stable) with
// TORCH_TARGET_VERSION pinned to 2.14, so one binary loads on every PyTorch
// from 2.14 on, instead of one wheel per torch minor. Nothing here may include
// ATen, c10 or torch/extension.h: those are the version-locked C++ ABI.
//
// nelux::Tensor is a torch::stable::Tensor (a shared handle to the same
// TensorImpl a Python torch.Tensor wraps) plus the handful of ordinary
// LibTorch conveniences the decoder and encoder use. Metadata calls go
// straight to C shims; ops go through torch's boxed dispatcher.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nelux
{

using ScalarType = torch::headeronly::ScalarType;
using Dtype = ScalarType;
using Layout = torch::headeronly::Layout;
using MemoryFormat = torch::headeronly::MemoryFormat;
using DeviceType = torch::headeronly::DeviceType;
using Device = torch::stable::Device;
using IntArrayRef = torch::headeronly::IntHeaderOnlyArrayRef;

inline constexpr ScalarType kUInt8 = ScalarType::Byte;
inline constexpr ScalarType kInt8 = ScalarType::Char;
inline constexpr ScalarType kInt16 = ScalarType::Short;
inline constexpr ScalarType kUInt16 = ScalarType::UInt16;
inline constexpr ScalarType kInt32 = ScalarType::Int;
inline constexpr ScalarType kUInt32 = ScalarType::UInt32;
inline constexpr ScalarType kInt64 = ScalarType::Long;
inline constexpr ScalarType kFloat16 = ScalarType::Half;
inline constexpr ScalarType kBFloat16 = ScalarType::BFloat16;
inline constexpr ScalarType kFloat32 = ScalarType::Float;
inline constexpr ScalarType kFloat64 = ScalarType::Double;
inline constexpr ScalarType kBool = ScalarType::Bool;
inline constexpr DeviceType kCPU = DeviceType::CPU;
inline constexpr DeviceType kCUDA = DeviceType::CUDA;
inline constexpr Layout kStrided = Layout::Strided;

namespace tensor_detail
{
// torch's own message for a failed shim call, without the C++ backtrace and
// without stable's stderr dump of it (STABLE_TORCH_ERROR_CODE_CHECK prints the
// full trace on every failure, including the ones a caller catches).
[[noreturn]] inline void throwTorchError(const char* what)
{
    const char* message = torch_exception_get_what_without_backtrace();
    throw std::runtime_error(message && *message ? std::string(message)
                                                 : std::string(what) + " failed");
}
} // namespace tensor_detail

#define NELUX_TORCH_CHECK(call)                                                \
    do                                                                         \
    {                                                                          \
        if ((call) != TORCH_SUCCESS)                                           \
            ::nelux::tensor_detail::throwTorchError(#call);                    \
    } while (0)

// The ordinary at::TensorOptions subset Nelux uses: dtype, device, pinning.
class TensorOptions
{
  public:
    TensorOptions() = default;
    TensorOptions(ScalarType dtype) : dtype_(dtype) {}
    TensorOptions(Device device) : device_(device) {}
    TensorOptions(DeviceType type) : device_(Device(type)) {}

    TensorOptions dtype(ScalarType dtype) const
    {
        TensorOptions o = *this;
        o.dtype_ = dtype;
        return o;
    }
    TensorOptions device(Device device) const
    {
        TensorOptions o = *this;
        o.device_ = device;
        return o;
    }
    TensorOptions device(DeviceType type, int index = -1) const
    {
        return device(Device(type, static_cast<torch::stable::DeviceIndex>(index)));
    }
    TensorOptions pinned_memory(bool pinned) const
    {
        TensorOptions o = *this;
        o.pinned_ = pinned;
        return o;
    }

    ScalarType dtype() const { return dtype_; }
    Device device() const { return device_; }
    bool pinned_memory() const { return pinned_; }

  private:
    ScalarType dtype_ = ScalarType::Float;
    Device device_ = Device(DeviceType::CPU);
    bool pinned_ = false;
};

class Tensor;
Tensor empty(IntArrayRef shape, const TensorOptions& options = {});

class Tensor : public torch::stable::Tensor
{
    using Base = torch::stable::Tensor;

  public:
    // An undefined tensor, like a default at::Tensor. The stable default
    // constructor allocates an uninitialized TensorImpl handle; a null handle
    // is cheaper and every query below treats it as undefined.
    Tensor() : Base(nullHandle()) {}
    Tensor(const Base& t) : Base(t) {}
    Tensor(Base&& t) noexcept : Base(std::move(t)) {}
    explicit Tensor(AtenTensorHandle handle) : Base(handle) {}

    bool defined() const { return get() != nullptr && Base::defined(); }
    // An undefined at::Tensor reports no elements and no data; so does this.
    int64_t numel() const { return get() ? Base::numel() : 0; }

    // ordinary .dtype() is a TypeMeta that compares equal to a ScalarType.
    ScalarType dtype() const { return scalar_type(); }
    TensorOptions options() const { return TensorOptions(scalar_type()).device(device()); }

    bool is_floating_point() const
    {
        switch (scalar_type())
        {
        case ScalarType::Half:
        case ScalarType::Float:
        case ScalarType::Double:
        case ScalarType::BFloat16:
        case ScalarType::Float8_e5m2:
        case ScalarType::Float8_e4m3fn:
        case ScalarType::Float8_e5m2fnuz:
        case ScalarType::Float8_e4m3fnuz:
            return true;
        default:
            return false;
        }
    }

    int64_t size(int64_t dim) const
    {
        const int64_t n = this->dim();
        return Base::size(dim < 0 ? dim + n : dim);
    }
    int64_t stride(int64_t dim) const
    {
        const int64_t n = this->dim();
        return Base::stride(dim < 0 ? dim + n : dim);
    }
    int64_t nbytes() const { return numel() * static_cast<int64_t>(element_size()); }

    void* data_ptr() const { return get() ? Base::data_ptr() : nullptr; }
    template <class T> T* data_ptr() const { return Base::mutable_data_ptr<T>(); }

    Tensor to(ScalarType dtype, bool non_blocking = false, bool copy = false) const;
    Tensor to(Device device, bool non_blocking = false, bool copy = false) const;
    Tensor to(DeviceType type, bool non_blocking = false, bool copy = false) const
    {
        return to(Device(type), non_blocking, copy);
    }
    Tensor to(Device device, ScalarType dtype, bool non_blocking = false,
              bool copy = false) const;
    Tensor to(const TensorOptions& options, bool non_blocking = false,
              bool copy = false) const
    {
        return to(options.device(), options.dtype(), non_blocking, copy);
    }
    Tensor cpu() const { return to(Device(DeviceType::CPU)); }

    // Like TensorBase::contiguous: an already-contiguous tensor is returned
    // as is, without a dispatcher round trip.
    Tensor contiguous() const;
    Tensor clone() const;
    Tensor detach() const;

    Tensor view(IntArrayRef shape) const;
    // Reinterpret the bytes as another dtype (Tensor.view(dtype)).
    Tensor view(ScalarType dtype) const;
    Tensor view(std::initializer_list<int64_t> shape) const
    {
        return view(IntArrayRef(shape.begin(), shape.size()));
    }
    Tensor reshape(IntArrayRef shape) const;
    Tensor reshape(std::initializer_list<int64_t> shape) const
    {
        return reshape(IntArrayRef(shape.begin(), shape.size()));
    }
    Tensor narrow(int64_t dim, int64_t start, int64_t length) const;
    Tensor select(int64_t dim, int64_t index) const;
    // t[i] on a tensor is select(0, i).
    Tensor operator[](int64_t index) const { return select(0, index); }
    Tensor permute(IntArrayRef dims) const;
    Tensor permute(std::initializer_list<int64_t> dims) const
    {
        return permute(IntArrayRef(dims.begin(), dims.size()));
    }
    Tensor squeeze(int64_t dim) const;
    Tensor unsqueeze(int64_t dim) const;
    Tensor flip(IntArrayRef dims) const;
    Tensor flip(std::initializer_list<int64_t> dims) const
    {
        return flip(IntArrayRef(dims.begin(), dims.size()));
    }

    const Tensor& copy_(const Tensor& source, bool non_blocking = false) const;

    Tensor round() const;
    // A lazily negated view (e.g. the .imag of a conjugated complex tensor):
    // its memory holds the negated values.
    bool is_neg() const;
    template <class Lo, class Hi> Tensor clamp(Lo minimum, Hi maximum) const;

  private:
    static Base nullHandle();
};

namespace tensor_detail
{
// A stack slot for a tensor. nelux::Tensor must be passed as its stable base:
// the generic StableIValue conversion only takes trivially copyable types.
inline StableIValue iv(const torch::stable::Tensor& t) { return torch::stable::detail::from(t); }
template <class T,
          std::enable_if_t<!std::is_base_of_v<torch::stable::Tensor, T>, int> = 0>
inline StableIValue iv(const T& v)
{
    return torch::stable::detail::from(v);
}

// One boxed dispatcher call; the result is the first stack slot.
template <class... Args>
inline StableIValue callRaw(const char* name, const char* overload, const Args&... args)
{
    std::array<StableIValue, sizeof...(Args)> stack{iv(args)...};
    NELUX_TORCH_CHECK(torch_call_dispatcher(name, overload, stack.data(), TORCH_ABI_VERSION));
    return stack[0];
}
template <class... Args>
inline Tensor call(const char* name, const char* overload, const Args&... args)
{
    return Tensor(torch::stable::detail::to<torch::stable::Tensor>(
        callRaw(name, overload, args...)));
}

// A Python-number operand for a Tensor overload. The stable dispatcher has no
// Scalar argument, so the number travels as a 0-dim tensor typed to promote
// exactly as the ordinary Scalar overload's wrapped number does: an integer
// keeps the tensor's dtype; a float keeps a floating tensor's dtype (and stays
// double so its value is unrounded) and makes an integral one float32, the
// default dtype.
template <class S>
inline Tensor scalarOperand(const Tensor& self, S value)
{
    static_assert(std::is_arithmetic_v<S>, "scalar operand must be arithmetic");
    AtenTensorHandle handle = nullptr;
    if constexpr (std::is_integral_v<S>)
        NELUX_TORCH_CHECK(aoti_torch_scalar_to_tensor_int64(static_cast<int64_t>(value), &handle));
    else if (self.is_floating_point())
        NELUX_TORCH_CHECK(aoti_torch_scalar_to_tensor_float64(static_cast<double>(value), &handle));
    else
        NELUX_TORCH_CHECK(aoti_torch_scalar_to_tensor_float32(static_cast<float>(value), &handle));
    return Tensor(handle);
}

template <class S>
inline Tensor binary(const char* name, const Tensor& self, S value)
{
    return call(name, "Tensor", self, scalarOperand(self, value));
}
} // namespace tensor_detail

inline Tensor::Base Tensor::nullHandle()
{
    // The handle's deleter does run for null; deleting a null handle is a
    // no-op in the shim.
    return Base(static_cast<AtenTensorHandle>(nullptr));
}

using OptionalDtype = std::optional<ScalarType>;
using OptionalLayout = std::optional<Layout>;
using OptionalDevice = std::optional<torch::stable::Device>;
using OptionalBool = std::optional<bool>;
using OptionalMemoryFormat = std::optional<MemoryFormat>;

inline Tensor Tensor::to(ScalarType dtype, bool non_blocking, bool copy) const
{
    if (!copy && scalar_type() == dtype)
        return *this;
    return tensor_detail::call("aten::to", "dtype_layout", *this, OptionalDtype(dtype),
                               OptionalLayout(), OptionalDevice(), OptionalBool(),
                               non_blocking, copy, OptionalMemoryFormat());
}
inline Tensor Tensor::to(Device device, bool non_blocking, bool copy) const
{
    if (!copy && this->device() == device)
        return *this;
    return tensor_detail::call("aten::to", "dtype_layout", *this, OptionalDtype(),
                               OptionalLayout(), OptionalDevice(device), OptionalBool(),
                               non_blocking, copy, OptionalMemoryFormat());
}
inline Tensor Tensor::to(Device device, ScalarType dtype, bool non_blocking, bool copy) const
{
    if (!copy && this->device() == device && scalar_type() == dtype)
        return *this;
    return tensor_detail::call("aten::to", "dtype_layout", *this, OptionalDtype(dtype),
                               OptionalLayout(), OptionalDevice(device), OptionalBool(),
                               non_blocking, copy, OptionalMemoryFormat());
}
inline Tensor Tensor::contiguous() const
{
    if (is_contiguous())
        return *this;
    return tensor_detail::call("aten::contiguous", "", *this, MemoryFormat::Contiguous);
}
inline Tensor Tensor::clone() const
{
    return tensor_detail::call("aten::clone", "", *this, OptionalMemoryFormat());
}
inline Tensor Tensor::detach() const { return tensor_detail::call("aten::detach", "", *this); }
inline Tensor Tensor::view(IntArrayRef shape) const
{
    return tensor_detail::call("aten::view", "", *this, shape);
}
inline Tensor Tensor::view(ScalarType dtype) const
{
    return tensor_detail::call("aten::view", "dtype", *this, dtype);
}
inline Tensor Tensor::reshape(IntArrayRef shape) const
{
    return tensor_detail::call("aten::reshape", "", *this, shape);
}
inline Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const
{
    return tensor_detail::call("aten::narrow", "", *this, dim, start, length);
}
inline Tensor Tensor::select(int64_t dim, int64_t index) const
{
    return tensor_detail::call("aten::select", "int", *this, dim, index);
}
inline Tensor Tensor::permute(IntArrayRef dims) const
{
    return tensor_detail::call("aten::permute", "", *this, dims);
}
inline Tensor Tensor::squeeze(int64_t dim) const
{
    return tensor_detail::call("aten::squeeze", "dim", *this, dim);
}
inline Tensor Tensor::unsqueeze(int64_t dim) const
{
    return tensor_detail::call("aten::unsqueeze", "", *this, dim);
}
inline Tensor Tensor::flip(IntArrayRef dims) const
{
    return tensor_detail::call("aten::flip", "", *this, dims);
}
inline const Tensor& Tensor::copy_(const Tensor& source, bool non_blocking) const
{
    tensor_detail::callRaw("aten::copy_", "", *this, source, non_blocking);
    return *this;
}
inline Tensor Tensor::round() const { return tensor_detail::call("aten::round", "", *this); }
inline bool Tensor::is_neg() const
{
    return torch::stable::detail::to<bool>(tensor_detail::callRaw("aten::is_neg", "", *this));
}

namespace tensor_detail
{
// clamp.Tensor, unlike the arithmetic ops, wants its bounds on self's device.
// A CUDA bound is filled from the CPU number by a kernel, without a host sync.
template <class S> inline Tensor boundOperand(const Tensor& self, S value)
{
    Tensor number = scalarOperand(self, value);
    if (self.is_cpu())
        return number;
    Tensor bound = empty(IntArrayRef(), TensorOptions(number.scalar_type()).device(self.device()));
    callRaw("aten::fill_", "Tensor", bound, number);
    return bound;
}
} // namespace tensor_detail

template <class Lo, class Hi> inline Tensor Tensor::clamp(Lo minimum, Hi maximum) const
{
    return tensor_detail::call(
        "aten::clamp", "Tensor", *this,
        std::optional<torch::stable::Tensor>(tensor_detail::boundOperand(*this, minimum)),
        std::optional<torch::stable::Tensor>(tensor_detail::boundOperand(*this, maximum)));
}

template <class S, std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
inline Tensor operator*(const Tensor& self, S value)
{
    return tensor_detail::binary("aten::mul", self, value);
}
template <class S, std::enable_if_t<std::is_arithmetic_v<S>, int> = 0>
inline Tensor operator/(const Tensor& self, S value)
{
    return tensor_detail::binary("aten::div", self, value);
}

// Contiguous row-major strides for @p shape.
inline std::vector<int64_t> contiguousStrides(IntArrayRef shape)
{
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i)
    {
        strides[i - 1] = stride;
        stride *= std::max<int64_t>(shape[i - 1], 1);
    }
    return strides;
}

// torch::empty. Contiguous, unpinned tensors go straight to the C allocator
// shim (one call, no boxed dispatch); a frame buffer is allocated per decoded
// frame on several paths.
inline Tensor empty(IntArrayRef shape, const TensorOptions& options)
{
    if (!options.pinned_memory())
    {
        std::array<int64_t, 8> inlineStrides{};
        std::vector<int64_t> heapStrides;
        int64_t* strides = inlineStrides.data();
        if (shape.size() > inlineStrides.size())
        {
            heapStrides.resize(shape.size());
            strides = heapStrides.data();
        }
        int64_t stride = 1;
        for (size_t i = shape.size(); i > 0; --i)
        {
            if (shape[i - 1] < 0)
                throw std::invalid_argument("Trying to create tensor with negative dimension " +
                                            std::to_string(shape[i - 1]));
            strides[i - 1] = stride;
            stride *= std::max<int64_t>(shape[i - 1], 1);
        }
        const Device device = options.device();
        AtenTensorHandle handle = nullptr;
        NELUX_TORCH_CHECK(aoti_torch_empty_strided(
            static_cast<int64_t>(shape.size()), shape.data(), strides,
            torch::stable::detail::to<int32_t>(torch::stable::detail::from(options.dtype())),
            torch::stable::detail::to<int32_t>(torch::stable::detail::from(device.type())),
            device.has_index() || device.type() == DeviceType::CPU
                ? device.index()
                : torch::stable::accelerator::getCurrentDeviceIndex(),
            &handle));
        return Tensor(handle);
    }
    return tensor_detail::call("aten::empty", "memory_format", shape,
                               std::optional<ScalarType>(options.dtype()), std::optional<Layout>(),
                               std::optional<torch::stable::Device>(options.device()),
                               std::optional<bool>(true), std::optional<MemoryFormat>());
}
inline Tensor empty(std::initializer_list<int64_t> shape, const TensorOptions& options = {})
{
    return empty(IntArrayRef(shape.begin(), shape.size()), options);
}
inline Tensor empty(const std::vector<int64_t>& shape, const TensorOptions& options = {})
{
    return empty(IntArrayRef(shape.data(), shape.size()), options);
}

// torch::from_blob over a contiguous buffer, with a deleter that runs once
// the last reference to the storage dies.
template <class F, std::enable_if_t<std::is_invocable_v<F, void*>, int> = 0>
inline Tensor from_blob(void* data, IntArrayRef shape, F deleter, const TensorOptions& options)
{
    const std::vector<int64_t> strides = contiguousStrides(shape);
    return torch::stable::from_blob(data, shape, IntArrayRef(strides.data(), strides.size()),
                                    options.device(), options.dtype(), std::move(deleter));
}

// torch::from_blob without a deleter: the tensor borrows @p data, which must
// outlive it.
inline Tensor from_blob(void* data, IntArrayRef shape, const TensorOptions& options)
{
    const std::vector<int64_t> strides = contiguousStrides(shape);
    return torch::stable::from_blob(data, shape, IntArrayRef(strides.data(), strides.size()),
                                    options.device(), options.dtype());
}
inline Tensor from_blob(void* data, std::initializer_list<int64_t> shape,
                        const TensorOptions& options)
{
    return from_blob(data, IntArrayRef(shape.begin(), shape.size()), options);
}
template <class F, std::enable_if_t<std::is_invocable_v<F, void*>, int> = 0>
inline Tensor from_blob(void* data, std::initializer_list<int64_t> shape, F deleter,
                        const TensorOptions& options)
{
    return from_blob(data, IntArrayRef(shape.begin(), shape.size()), std::move(deleter), options);
}

inline Tensor rot90(const Tensor& self, int64_t turns, IntArrayRef dims)
{
    return tensor_detail::call("aten::rot90", "", self, turns, dims);
}
inline Tensor rot90(const Tensor& self, int64_t turns, std::initializer_list<int64_t> dims)
{
    return rot90(self, turns, IntArrayRef(dims.begin(), dims.size()));
}

inline bool equal(const Tensor& a, const Tensor& b)
{
    return torch::stable::detail::to<bool>(tensor_detail::callRaw("aten::equal", "", a, b));
}

inline std::string toString(ScalarType type) { return torch::headeronly::toString(type); }

inline int64_t elementSize(ScalarType type)
{
    return static_cast<int64_t>(aoti_torch_dtype_element_size(
        torch::stable::detail::to<int32_t>(torch::stable::detail::from(type))));
}

// at::get_num_threads: the size of torch's intra-op pool.
inline int getNumThreads() { return static_cast<int>(torch::stable::get_num_threads()); }

#ifdef NELUX_ENABLE_CUDA
// The caller's current stream on @p device, as torch.cuda.current_stream()
// would report it, through the device-generic stable accelerator API.
inline void* currentStreamHandle(int device)
{
    return torch::stable::accelerator::getCurrentStream(
               static_cast<torch::stable::accelerator::DeviceIndex>(device))
        .nativeHandle();
}
#endif

} // namespace nelux
