#pragma once
// pybind11 <-> torch.Tensor conversion for nelux::Tensor, through the stable
// ABI's official Python interop shims (torch/csrc/stable/pyobject.h, 2.14+).
// This replaces the caster torch/extension.h used to provide; the result is
// the same Python object torch's own bindings would produce (it shares the
// TensorImpl, and a tensor that came from Python returns as that object).

#include <CudaRuntimeGuard.hpp>
#include <Tensor.hpp>

#include <pybind11/pybind11.h>
#include <torch/csrc/stable/pyobject.h>

#include <vector>

namespace nelux::python
{
// torch._C.TensorBase, the base of every torch.Tensor (and subclass).
// Resolved once at module import, with the GIL held.
inline PyTypeObject*& tensorBaseType()
{
    static PyTypeObject* type = nullptr;
    return type;
}

// torch's Python exception for each TorchError kind (null: RuntimeError).
inline PyObject*& torchExceptionType(TorchError::Kind kind)
{
    static PyObject* types[5] = {};
    return types[static_cast<int>(kind)];
}

inline void initTensorInterop()
{
    namespace py = pybind11;
    py::module_ torch = py::module_::import("torch");
    py::object base = py::module_::import("torch._C").attr("TensorBase");
    tensorBaseType() = reinterpret_cast<PyTypeObject*>(base.release().ptr());
    // Kept for the life of the process: the module never unloads.
    auto keep = [&](TorchError::Kind kind, const char* name) {
        if (py::hasattr(torch, name))
            torchExceptionType(kind) = py::object(torch.attr(name)).release().ptr();
    };
    keep(TorchError::Kind::OutOfMemory, "OutOfMemoryError");
    keep(TorchError::Kind::Accelerator, "AcceleratorError");
    torchExceptionType(TorchError::Kind::NotImplemented) = PyExc_NotImplementedError;
    torchExceptionType(TorchError::Kind::Index) = PyExc_IndexError;
    // CPU-only and ROCm torch report no CUDA version; Nelux's GPU paths are
    // CUDA, so they turn those builds away up front.
    nelux::setTorchHasCuda(!torch.attr("version").attr("cuda").is_none());
}

// A TorchError as the exception torch's own bindings raise for it.
inline void registerTorchErrorTranslator()
{
    pybind11::register_local_exception_translator([](std::exception_ptr p) {
        try
        {
            if (p)
                std::rethrow_exception(p);
        }
        catch (const TorchError& e)
        {
            PyObject* type = torchExceptionType(e.kind());
            PyErr_SetString(type ? type : PyExc_RuntimeError, e.what());
        }
    });
}

inline bool isTensor(PyObject* obj)
{
    PyTypeObject* base = tensorBaseType();
    return base && PyObject_TypeCheck(obj, base);
}

#ifdef NELUX_ENABLE_CUDA
// frame.record_stream(torch.cuda.current_stream(frame.device)), with the GIL
// held. torch's record_stream has no stable-ABI form (the dispatcher takes no
// Stream argument), so it goes through torch's Python binding: the same
// caching-allocator call, about 0.2 us. It marks the storage itself, as
// before, so no in-place change to the tensor (set_, resize_, a storage
// resize) can free memory a queued kernel on that stream still reads.
inline void recordCurrentStream(const nelux::Tensor& frame)
{
    namespace py = pybind11;
    const int device = static_cast<int>(frame.device().index());
    void* handle = currentStreamHandle(device);

    // torch.cuda.Stream objects for the stream handles seen, per device. A
    // handle names one stream for the life of the process (torch's pooled
    // streams are never destroyed; an ExternalStream's id is its handle).
    struct Known
    {
        int device;
        void* handle;
        PyObject* stream;
    };
    static std::vector<Known> known;
    PyObject* stream = nullptr;
    py::object fresh;
    for (const Known& k : known)
        if (k.device == device && k.handle == handle)
        {
            stream = k.stream;
            break;
        }
    if (!stream)
    {
        static PyObject* currentStream =
            py::object(py::module_::import("torch.cuda").attr("current_stream")).release().ptr();
        py::object s = py::reinterpret_borrow<py::object>(currentStream)(device);
        if (s.attr("cuda_stream").cast<uintptr_t>() != reinterpret_cast<uintptr_t>(handle))
            throw std::runtime_error("torch.cuda.current_stream() disagrees with torch's "
                                     "current stream handle");
        // Bounded, and never evicted (another thread may be using an entry
        // while the call below runs Python): past 64 streams, a process
        // cycling through external streams looks each new one up per call.
        if (known.size() < 64)
        {
            stream = s.release().ptr();
            known.push_back({device, handle, stream});
        }
        else
        {
            fresh = std::move(s);
            stream = fresh.ptr();
        }
    }

    static PyObject* name = PyUnicode_InternFromString("record_stream");
    py::object tensor =
        py::reinterpret_steal<py::object>(static_cast<PyObject*>(torch::stable::tensor_to_pyobject(frame)));
    PyObject* result = PyObject_CallMethodOneArg(tensor.ptr(), name, stream);
    if (!result)
        throw py::error_already_set();
    Py_DECREF(result);
}
#endif
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
