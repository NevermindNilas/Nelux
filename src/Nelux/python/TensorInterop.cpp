#include <TensorInterop.hpp>
#include <torch/csrc/stable/library.h>
#include <optional>
#include <stdexcept>

namespace nelux::python {
namespace {
using torch::stable::Tensor;
enum class Direction { Capture, Export };
struct Exchange;
thread_local Exchange* active = nullptr;
struct Exchange {
    Direction direction;
    Exchange* previous;
    std::optional<Tensor> tensor;
    bool used = false;
    explicit Exchange(Direction d) : direction(d), previous(active) { active = this; }
    ~Exchange() { active = previous; }
    Exchange(const Exchange&) = delete;
    Exchange& operator=(const Exchange&) = delete;
};
Exchange& requireExchange(Direction direction) {
    if (!active || active->direction != direction || active->used)
        throw std::runtime_error("Nelux's private tensor exchange operator requires an active conversion");
    active->used = true;
    return *active;
}
void capture(Tensor tensor) {
    requireExchange(Direction::Capture).tensor.emplace(std::move(tensor));
}
Tensor exportTensor() {
    auto& exchange = requireExchange(Direction::Export);
    if (!exchange.tensor) throw std::runtime_error("Empty Nelux tensor export context");
    return std::move(*exchange.tensor);
}
void requireGil() {
    if (!PyGILState_Check()) throw std::runtime_error("Nelux tensor conversion requires the Python GIL");
}
struct PythonBoundary {
    pybind11::object tensor_type;
    pybind11::object capture_op;
    pybind11::object export_op;
    pybind11::object disable_function;
    pybind11::object enter_function;
    pybind11::object exit_function;
    pybind11::object disable_dispatch;
    pybind11::object enter_dispatch;
    pybind11::object exit_dispatch;
    PythonBoundary();
};
PythonBoundary& pythonBoundary() {
    // Interpreter-owned Python references survive sys.modules eviction and
    // retained Reader objects. CPython destroys the capsule with the GIL on
    // interpreter shutdown. No process-global PyObject or native tensor registry
    // exists, and a second interpreter cannot reuse the first one's objects.
    constexpr auto key = "nelux.stable_tensor_boundary";
    constexpr auto capsule_name = "nelux.stable_tensor_boundary.owner";
    auto* dictionary = PyInterpreterState_GetDict(PyInterpreterState_Get());
    auto* cached = PyDict_GetItemString(dictionary, key);
    if (!cached && PyErr_Occurred()) throw pybind11::error_already_set();
    if (!cached) {
        auto boundary = std::make_unique<PythonBoundary>();
        pybind11::capsule owner(boundary.get(), capsule_name, [](void* opaque) {
            delete static_cast<PythonBoundary*>(opaque);
        });
        // The capsule owns the boundary even if the dictionary insertion fails.
        boundary.release();
        if (PyDict_SetItemString(dictionary, key, owner.ptr()) < 0)
            throw pybind11::error_already_set();
        cached = owner.ptr();
    }
    auto* pointer = PyCapsule_GetPointer(cached, capsule_name);
    if (!pointer) throw pybind11::error_already_set();
    return *static_cast<PythonBoundary*>(pointer);
}
class DisablePythonTensorOverrides {
    PythonBoundary& boundary;
    pybind11::object function_scope;
    pybind11::object dispatch_scope;
public:
    explicit DisablePythonTensorOverrides(PythonBoundary& cache) : boundary(cache) {
        function_scope = boundary.disable_function();
        boundary.enter_function(function_scope);
        try {
            dispatch_scope = boundary.disable_dispatch();
            boundary.enter_dispatch(dispatch_scope);
        } catch (...) {
            boundary.exit_function(function_scope, pybind11::none(), pybind11::none(), pybind11::none());
            throw;
        }
    }
    ~DisablePythonTensorOverrides() {
        // These built-in scopes only restore thread-local dispatch flags; they
        // do not invoke user __exit__ hooks. Keep destructors exception-safe.
        try {
            boundary.exit_dispatch(dispatch_scope, pybind11::none(), pybind11::none(), pybind11::none());
            boundary.exit_function(function_scope, pybind11::none(), pybind11::none(), pybind11::none());
        } catch (pybind11::error_already_set& error) {
            error.discard_as_unraisable("Restoring Nelux tensor conversion scopes");
        }
    }
};

PythonBoundary::PythonBoundary() {
    auto torch = pybind11::module_::import("torch");
    auto operators = torch.attr("ops").attr("nelux_abi");
    capture_op = operators.attr("_capture");
    export_op = operators.attr("_export");
    auto core = torch.attr("_C");
    disable_function = core.attr("DisableTorchFunction");
    enter_function = disable_function.attr("__enter__");
    exit_function = disable_function.attr("__exit__");
    disable_dispatch = core.attr("_DisableTorchDispatch");
    enter_dispatch = disable_dispatch.attr("__enter__");
    exit_dispatch = disable_dispatch.attr("__exit__");

    // torch.Tensor is a mutable module attribute. Derive the canonical Python
    // type from an actual stable export, so rebinding that attribute cannot
    // reject genuine tensors or admit unrelated objects to the unpacker.
    // This one-time exchange uses the same guards and native wrapping as later
    // conversions, without private Tensor layouts or Python class symbols.
    Exchange exchange(Direction::Export);
    DisablePythonTensorOverrides overrides(*this);
    exchange.tensor.emplace(nelux::tensor::empty({0}, nelux::tensor::kUInt8));
    auto exemplar = export_op();
    if (!exchange.used)
        throw std::runtime_error("PyTorch did not initialize Nelux's tensor conversion");
    tensor_type = pybind11::reinterpret_borrow<pybind11::object>(
        reinterpret_cast<PyObject*>(Py_TYPE(exemplar.ptr())));
}
} // namespace

void initializeTensorInterop() { requireGil(); (void)pythonBoundary(); }

STABLE_TORCH_LIBRARY(nelux_abi, m) {
    m.def("_capture(Tensor tensor) -> ()");
    m.def("_export() -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(nelux_abi, CompositeExplicitAutograd, m) {
    m.impl("_capture", TORCH_BOX(&capture));
    m.impl("_export", TORCH_BOX(&exportTensor));
}

bool loadTensor(pybind11::handle source, Tensor& output) {
    requireGil();
    auto& boundary = pythonBoundary();
    // Python isinstance() consults a user object's __class__ property. A fake
    // Tensor can pass that check and crash PyTorch's dispatcher unpacker.
    // Only actual Tensor subclasses may reach that unpacker.
    if (!PyObject_TypeCheck(source.ptr(),
            reinterpret_cast<PyTypeObject*>(boundary.tensor_type.ptr()))) return false;
    Exchange exchange(Direction::Capture);
    DisablePythonTensorOverrides overrides(boundary);
    boundary.capture_op(source);
    if (!exchange.used || !exchange.tensor)
        throw std::runtime_error("PyTorch did not complete Nelux's tensor conversion");
    output = std::move(*exchange.tensor);
    return true;
}
pybind11::object castTensor(const Tensor& tensor) {
    requireGil();
    if (!tensor.defined()) return pybind11::none();
    Exchange exchange(Direction::Export);
    exchange.tensor.emplace(tensor);
    auto& boundary = pythonBoundary();
    DisablePythonTensorOverrides overrides(boundary);
    auto output = boundary.export_op();
    if (!exchange.used) throw std::runtime_error("PyTorch did not complete Nelux's tensor export");
    return output;
}
} // namespace nelux::python
