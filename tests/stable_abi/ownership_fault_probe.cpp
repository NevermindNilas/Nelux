// Test-only shim interposition: production contains no allocation fault hooks.
// Include the real stable Tensor first so its control-block failure cleanup
// still calls the real native tensor deleter.
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <array>
#include <cstdlib>
#include <iostream>
#include <new>
#include <optional>
#include <stdexcept>
#include <thread>

extern "C" AOTITorchError fault_from_blob(
    void*, int64_t, const int64_t*, const int64_t*, int64_t, int32_t, int32_t,
    int32_t, AtenTensorHandle*, int32_t, const uint8_t*, int64_t,
    void (*)(void*, void*), void*);
#define torch_from_blob fault_from_blob
#include <TensorSupport.hpp>
#undef torch_from_blob

namespace fault {
bool tracking = false;
int fail_after = -1;
size_t allocations = 0;
std::array<std::atomic<void*>, 256> live{};
enum class Mode { Success, ErrorNoCallback, CallbackThenError, WrapperAllocationFailure };
Mode mode = Mode::Success;
int shim_calls = 0;
int64_t rank = 0;
std::array<int64_t, 32> observed_strides{};
void (*callback)(void*, void*) = nullptr;
void* callback_context = nullptr;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void record(void* pointer) {
    if (!tracking) return;
    ++allocations;
    for (auto& slot : live) {
        void* expected = nullptr;
        if (slot.compare_exchange_strong(expected, pointer, std::memory_order_relaxed)) return;
    }
    std::abort();
}
void forget(void* pointer) {
    for (auto& slot : live) {
        void* expected = pointer;
        if (slot.compare_exchange_strong(expected, nullptr, std::memory_order_relaxed)) return;
    }
}
void reset(Mode next = Mode::Success) {
    for (const auto& pointer : live) require(pointer.load() == nullptr, "A tracked allocation survived its test");
    mode = next;
    fail_after = -1;
    allocations = 0;
    shim_calls = 0;
    callback = nullptr;
    callback_context = nullptr;
    tracking = true;
}
struct Counts { int releases = 0; int original_releases = 0; int context_releases = 0; int live_copies = 0; size_t release_allocations = 0; bool fail_copy = false; bool fail_release = false; };
void* data();
struct Lease {
    Counts* counts;
    bool copied = false;
    bool reentrant = false;
    Counts* nested_counts = nullptr;
    explicit Lease(Counts& counts, bool reentrant = false, Counts* nested_counts = nullptr)
        : counts(&counts), reentrant(reentrant), nested_counts(nested_counts) {}
    Lease(const Lease& other) : counts(other.counts), copied(true), reentrant(other.reentrant), nested_counts(other.nested_counts) {
        if (counts->fail_copy) throw std::runtime_error("Injected Context deleter-copy failure");
        ++counts->live_copies;
    }
    Lease(Lease&& other) noexcept : counts(other.counts), copied(other.copied), reentrant(other.reentrant), nested_counts(other.nested_counts) { other.counts = nullptr; }
    ~Lease() { if (counts && copied) --counts->live_copies; }
    void operator()(void* pointer) const {
        counts->release_allocations = allocations;
        require(++counts->releases == 1, "A native lease was released twice");
        if (copied) ++counts->context_releases;
        else ++counts->original_releases;
        if (reentrant) callback(pointer, callback_context);
        if (nested_counts) {
            auto nested = nelux::tensor::from_blob(data(), {2, 3}, Lease(*nested_counts), nelux::tensor::kUInt8);
        }
        std::free(pointer);
        if (counts->fail_release) throw std::runtime_error("Injected null-data deleter failure");
    }
};
void* data() {
    auto* pointer = std::malloc(6);
    if (!pointer) throw std::bad_alloc();
    auto* bytes = static_cast<uint8_t*>(pointer);
    for (int index = 0; index < 6; ++index) bytes[index] = static_cast<uint8_t>(index + 1);
    return pointer;
}
template<class Exception, class F> void fails(F operation) {
    try { operation(); }
    catch (const Exception&) { return; }
    throw std::runtime_error("Expected injected construction failure");
}
void complete(const Counts& counts, const char* name) {
    tracking = false;
    require(counts.releases == 1, "A failed or destroyed tensor leaked its lease");
    require(counts.live_copies == 0, "A deleter Context survived its final reference");
    for (const auto& pointer : live) require(pointer.load() == nullptr, "A Context or control block leaked");
    std::cout << "PASS " << name << '\n';
}
} // namespace fault

void* operator new(std::size_t size) {
    if (fault::fail_after == 0) { fault::fail_after = -1; throw std::bad_alloc(); }
    if (fault::fail_after > 0) --fault::fail_after;
    auto* pointer = std::malloc(size ? size : 1);
    if (!pointer) throw std::bad_alloc();
    fault::record(pointer);
    return pointer;
}
void operator delete(void* pointer) noexcept { fault::forget(pointer); std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { ::operator delete(pointer); }

extern "C" AOTITorchError fault_from_blob(
    void* data, int64_t ndim, const int64_t* sizes, const int64_t* strides,
    int64_t offset, int32_t dtype, int32_t device_type, int32_t device_index,
    AtenTensorHandle* output, int32_t layout, const uint8_t* metadata, int64_t metadata_size,
    void (*deleter)(void*, void*), void* context) {
    using namespace fault;
    ++shim_calls;
    rank = ndim;
    require(ndim <= static_cast<int64_t>(observed_strides.size()), "Test rank exceeds its fixed observation buffer");
    for (int64_t dimension = 0; dimension < ndim; ++dimension) observed_strides[dimension] = strides[dimension];
    callback = deleter;
    callback_context = context;
    if (mode == Mode::ErrorNoCallback) return AOTI_TORCH_FAILURE;
    if (mode == Mode::CallbackThenError) { deleter(data, context); return AOTI_TORCH_FAILURE; }
    // The successful paths use actual Torch storage and the actual stable
    // Tensor class, including destruction through retained native views.
    const bool previous_tracking = tracking;
    tracking = false;
    const auto error = torch_from_blob(data, ndim, sizes, strides, offset, dtype,
        device_type, device_index, output, layout, metadata, metadata_size, deleter, context);
    tracking = previous_tracking;
    if (error == AOTI_TORCH_SUCCESS && mode == Mode::WrapperAllocationFailure) fail_after = 0;
    return error;
}

int main() {
    using namespace fault;
    using namespace nelux::tensor;
    try {
        {
            reset(); Counts counts;
            fail_after = 0;
            fails<std::bad_alloc>([&] { from_blob(data(), {2, 3}, Lease(counts), kUInt8); });
            require(shim_calls == 0 && counts.original_releases == 1, "Context-allocation failure crossed the shim or lost the original deleter");
            complete(counts, "Context allocation failure");
        }
        {
            reset(); Counts counts; counts.fail_copy = true;
            fails<std::runtime_error>([&] { from_blob(data(), {2, 3}, Lease(counts), kUInt8); });
            require(shim_calls == 0 && counts.original_releases == 1, "Context-copy failure lost the original deleter");
            complete(counts, "Context deleter-copy failure");
        }
        for (const auto shape : {std::vector<int64_t>{-1}, std::vector<int64_t>{INT64_MAX, 2}}) {
            reset(); Counts counts;
            fails<std::invalid_argument>([&] { from_blob(data(), shape, Lease(counts), kUInt8); });
            require(shim_calls == 0 && counts.original_releases == 1, "Shape validation did not release its pending callback reference");
            complete(counts, "Pre-call shape validation");
        }
        {
            std::array<int64_t, 9> shape; shape.fill(1);
            reset(); Counts counts; fail_after = 1;
            fails<std::bad_alloc>([&] { from_blob(data(), shape, Lease(counts), kUInt8); });
            require(shim_calls == 0, "Stride allocation failed after crossing the shim");
            complete(counts, "Large-rank stride allocation failure");
        }
        for (auto mode : {Mode::ErrorNoCallback, Mode::CallbackThenError}) {
            reset(mode); Counts counts;
            fails<std::runtime_error>([&] { from_blob(data(), {2, 3}, Lease(counts), kUInt8); });
            require(shim_calls == 1, "Shim-error fault was not reached");
            require((mode == Mode::ErrorNoCallback ? counts.original_releases : counts.context_releases) == 1,
                "Shim error used the wrong cleanup owner");
            complete(counts, mode == Mode::ErrorNoCallback ? "Shim error without callback" : "Shim callback followed by error");
        }
        {
            reset(Mode::WrapperAllocationFailure); Counts counts;
            fails<std::bad_alloc>([&] { from_blob(data(), {2, 3}, Lease(counts), kUInt8); });
            require(shim_calls == 1 && counts.context_releases == 1 && counts.original_releases == 0,
                "Tensor control-block failure did not clean up through native storage");
            complete(counts, "Stable Tensor control-block allocation failure");
        }
        {
            reset(); Counts counts;
            std::optional<Tensor> tensor = from_blob(data(), {2, 3}, Lease(counts), kUInt8);
            require(allocations == 2, "Common frame wrapping did not use exactly one Context and one stable Tensor control block");
            require(rank == 2 && observed_strides[0] == 3 && observed_strides[1] == 1, "Inline contiguous strides changed");
            tracking = false;
            std::optional<Tensor> view = select(*tensor, 0, 0);
            tensor.reset();
            require(counts.releases == 0 && counts.live_copies == 1, "A retained native view lost its pool lease");
            require(view->const_data_ptr<uint8_t>()[2] == 3, "Retained view data changed");
            view.reset();
            complete(counts, "Delayed native storage destruction through a view");
        }
        {
            reset(); Counts counts;
            { auto tensor = from_blob(data(), {2, 3}, Lease(counts, true), kUInt8); }
            complete(counts, "Reentrant native deleter");
        }
        {
            reset(); Counts counts; Counts nested_counts;
            { auto tensor = from_blob(data(), {2, 3}, Lease(counts, false, &nested_counts), kUInt8); }
            require(nested_counts.releases == 1 && nested_counts.context_releases == 1 && nested_counts.live_copies == 0,
                "A nested from_blob lease did not remain independent of its outer deleter");
            complete(counts, "Native deleter constructing a distinct nested tensor");
        }
        {
            reset(); Counts counts;
            auto tensor = from_blob(data(), {2, 3}, Lease(counts), kUInt8);
            tracking = false;
            std::thread final_owner([tensor = std::move(tensor)] {});
            final_owner.join();
            complete(counts, "Cross-thread final native storage destruction");
        }
        {
            reset(); Counts counts;
            auto tensor = from_blob(nullptr, {2, 3}, Lease(counts), kUInt8);
            require(tensor.defined() && tensor.numel() == 6 && tensor.data_ptr() != nullptr,
                "Null input did not preserve the shim's empty_strided result");
            require(counts.original_releases == 1 && counts.context_releases == 0 && counts.live_copies == 0,
                "Null success retained a callback that native storage never owns");
            tracking = false;
            // The Tensor control block is still live, while its deleter Context
            // is already gone; destroy it before checking tracked allocations.
            { auto owner = std::move(tensor); }
            complete(counts, "Null-data success without native callback");
        }
        {
            reset(Mode::ErrorNoCallback); Counts counts;
            fails<std::runtime_error>([&] { from_blob(nullptr, {2, 3}, Lease(counts), kUInt8); });
            require(counts.original_releases == 1 && counts.context_releases == 0,
                "Null-data shim error lost its original deleter");
            complete(counts, "Null-data shim error without callback");
        }
        {
            reset(Mode::WrapperAllocationFailure); Counts counts;
            fails<std::bad_alloc>([&] { from_blob(nullptr, {2, 3}, Lease(counts), kUInt8); });
            require(counts.original_releases == 1 && counts.context_releases == 0,
                "Null-data wrapper allocation failure released its lease twice");
            complete(counts, "Null-data Tensor control-block allocation failure");
        }
        {
            reset(); Counts counts; counts.fail_release = true;
            fails<std::runtime_error>([&] { from_blob(nullptr, {2, 3}, Lease(counts), kUInt8); });
            require(counts.original_releases == 1 && counts.context_releases == 0,
                "Throwing null-data deleter did not use the original cleanup owner");
            require(counts.release_allocations == 2,
                "Null-data deleter ran before the stable Tensor owned its native handle");
            complete(counts, "Null-data deleter throwing after native storage ownership");
        }
        {
            std::array<int64_t, 9> shape; shape.fill(1);
            reset(); Counts counts;
            { auto tensor = from_blob(data(), shape, Lease(counts), kUInt8);
              require(rank == 9, "Large-rank vector fallback was not used");
              for (int dimension = 0; dimension < 9; ++dimension) require(observed_strides[dimension] == 1, "Large-rank contiguous strides changed"); }
            complete(counts, "Successful large-rank fallback");
        }
        return 0;
    } catch (const std::exception& error) {
        tracking = false;
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
