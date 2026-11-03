#pragma once
// pybind11 <-> torch.Tensor conversion for nelux::Tensor, through the stable
// ABI's official Python interop shims (torch/csrc/stable/pyobject.h, 2.14+).
// This replaces the caster torch/extension.h used to provide; the result is
// the same Python object torch's own bindings would produce (it shares the
// TensorImpl, and a tensor that came from Python returns as that object).

#include <Tensor.hpp>

#include <pybind11/pybind11.h>
#include <torch/csrc/stable/pyobject.h>

namespace nelux::python
{
// torch._C.TensorBase, the base of every torch.Tensor (and subclass).
// Resolved once at module import, with the GIL held.
inline PyTypeObject*& tensorBaseType()
{
    static PyTypeObject* type = nullptr;
    return type;
}

inline void initTensorInterop()
{
    namespace py = pybind11;
    py::object base = py::module_::import("torch._C").attr("TensorBase");
    tensorBaseType() = reinterpret_cast<PyTypeObject*>(base.release().ptr());
}

inline bool isTensor(PyObject* obj)
{
    PyTypeObject* base = tensorBaseType();
    return base && PyObject_TypeCheck(obj, base);
}
} // namespace nelux::python

namespace pybind11::detail
{
template <> struct type_caster<nelux::Tensor>
{
    PYBIND11_TYPE_CASTER(nelux::Tensor, const_name("torch.Tensor"));

    bool load(handle src, bool)
    {
        if (!src || !nelux::python::isTensor(src.ptr()))
            return false;
        value = nelux::Tensor(torch::stable::tensor_from_pyobject(src.ptr()));
        return true;
    }

    static handle cast(const nelux::Tensor& t, return_value_policy, handle)
    {
        // An undefined tensor is None, as with torch's own caster.
        if (!t.defined())
            return none().release();
        return handle(static_cast<PyObject*>(torch::stable::tensor_to_pyobject(t)));
    }
};
} // namespace pybind11::detail
