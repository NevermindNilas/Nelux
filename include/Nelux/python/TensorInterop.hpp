#pragma once
#include <TensorSupport.hpp>
#include <pybind11/pybind11.h>

namespace nelux::python {
void initializeTensorInterop();
// Both conversions require the GIL. Exchange uses the supported 2.12 stable
// dispatcher; no private Python Tensor layout or ordinary LibTorch caster.
bool loadTensor(pybind11::handle source, torch::stable::Tensor& output);
pybind11::object castTensor(const torch::stable::Tensor& tensor);
}

namespace pybind11::detail {
template<> struct type_caster<torch::stable::Tensor> {
    PYBIND11_TYPE_CASTER(torch::stable::Tensor, const_name("torch.Tensor"));
    bool load(handle source, bool) { return nelux::python::loadTensor(source, value); }
    static handle cast(const torch::stable::Tensor& tensor, return_value_policy, handle) {
        return nelux::python::castTensor(tensor).release();
    }
};
}
