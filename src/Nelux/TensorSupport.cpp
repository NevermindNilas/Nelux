#include <TensorSupport.hpp>
#include <array>

namespace nelux::tensor {
namespace {
template<class... Args>
Tensor dispatch(const char* name, const char* overload, const Args&... args) {
    std::array<StableIValue, sizeof...(Args)> values{torch::stable::detail::from(args)...};
    TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(name, overload, values.data(), TORCH_ABI_VERSION));
    return torch::stable::detail::to<Tensor>(values[0]);
}
Tensor integer(int64_t value) {
    AtenTensorHandle handle = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_scalar_to_tensor_int64(value, &handle));
    return Tensor(handle);
}
Tensor floating(const Tensor& input, double value) {
    if (is_floating_point(input)) {
        AtenTensorHandle handle = nullptr;
        TORCH_ERROR_CODE_CHECK(aoti_torch_scalar_to_tensor_float64(value, &handle));
        return Tensor(handle);
    }
    // An integral input combined with a floating Scalar promotes to torch's
    // current default floating dtype, including torch.set_default_dtype().
    auto scalar = torch::stable::empty({});
    return torch::stable::fill_(scalar, value);
}
Tensor scalar_binary(const Tensor& input, const Tensor& scalar, const char* op) {
    // TensorIterator gives dimensioned tensors precedence over CPU scalars.
    // A wrapped Scalar has that behavior even when input itself is 0-dimensional.
    const bool scalar_input = input.dim() == 0;
    auto source = scalar_input ? view(input, {1}) : input;
    auto output = dispatch(op, "Tensor", source, scalar);
    return scalar_input ? view(output, {}) : output;
}
Tensor clamp_bound(const Tensor& input, int64_t value) {
    // Signed integer Scalars permit the unsigned range [-max, max], whereas
    // fill_.Scalar's double argument rejects every negative unsigned value.
    const auto dtype = input.scalar_type();
    int64_t lo = std::numeric_limits<int64_t>::min(), hi = std::numeric_limits<int64_t>::max();
    if (dtype == kUInt8) { lo = -255; hi = 255; }
    else if (dtype == kInt8) { lo = -128; hi = 127; }
    else if (dtype == kInt16) { lo = -32768; hi = 32767; }
    else if (dtype == kInt32) { lo = std::numeric_limits<int32_t>::min(); hi = std::numeric_limits<int32_t>::max(); }
    else if (dtype == kFloat16 && input.is_cpu()) { lo = -65504; hi = 65504; }
    if (value < lo || value > hi) throw std::runtime_error("Clamp bound cannot be converted to the input dtype without overflow");
    if (dtype == kUInt8 || dtype == kInt8 || dtype == kInt16 || dtype == kInt32 || dtype == kInt64) {
        if (input.is_cpu()) return to(integer(value), dtype);
        if (value < -(INT64_C(1) << 53) || value > (INT64_C(1) << 53))
            return nelux::tensor::to(to(integer(value), dtype), input.device());
    }
    auto bound = empty({}, input.scalar_type(), input.device());
    // fill_.Scalar uses a direct versioned C shim and the same checked scalar
    // conversion as clamp.Scalar. In particular int8 upper=255 must throw.
    double converted = static_cast<double>(dtype == kUInt8 ? static_cast<uint8_t>(value) : value);
    // CUDA clamp performs half comparisons in float opmath and casts the
    // result back to half; a finite bound beyond half's range is valid there.
    if (dtype == kFloat16 && input.is_cuda() && (value < -65504 || value > 65504))
        converted = value < 0 ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    return torch::stable::fill_(bound, converted);
}
} // namespace
Tensor permute(const Tensor& t, Shape dims) { return dispatch("aten::permute", "", t, dims); }
Tensor stack(const std::vector<Tensor>& tensors, int64_t dim) { return dispatch("aten::stack", "", tensors, dim); }
Tensor index_select(const Tensor& t, int64_t dim, const Tensor& indices) { return dispatch("aten::index_select", "", t, dim, indices); }
Tensor arange(int64_t end, Dtype dtype, Device device) {
    if (end < 0) throw std::invalid_argument("arange end must be non-negative");
    auto indices = empty({end}, kInt64);
    auto* ptr = indices.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < end; ++i) ptr[i] = i;
    return to(indices, device, dtype);
}
Tensor round(const Tensor& t) { return dispatch("aten::round", "", t); }
Tensor clamp(const Tensor& t, int64_t minimum, int64_t maximum) {
    auto input = t.scalar_type() == Dtype::Bool ? to(t, kInt64) : t;
    return dispatch("aten::clamp", "Tensor", input,
        std::optional<Tensor>(clamp_bound(input, minimum)), std::optional<Tensor>(clamp_bound(input, maximum)));
}
Tensor mul_integer(const Tensor& t, int64_t value) { return scalar_binary(t, integer(value), "aten::mul"); }
Tensor mul_floating(const Tensor& t, double value) { return scalar_binary(t, floating(t, value), "aten::mul"); }
Tensor div_integer(const Tensor& t, int64_t value) { return scalar_binary(t, integer(value), "aten::div"); }
Tensor div_floating(const Tensor& t, double value) { return scalar_binary(t, floating(t, value), "aten::div"); }
} // namespace nelux::tensor
