"""Run the stable-only test extension without importing Nelux or its codecs."""
import gc
import sys

import torch

if len(sys.argv) != 2:
    raise SystemExit("Usage: check_boundary.py EXTENSION_DIRECTORY")
sys.path.insert(0, sys.argv[1])
import abi_probe as probe


def check_exchange():
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    class SpoofedTensor:
        @property
        def __class__(self):
            return torch.Tensor

    spoofed = SpoofedTensor()
    assert isinstance(spoofed, torch.Tensor)
    try:
        probe.identity(spoofed)
    except TypeError:
        pass
    else:
        raise AssertionError("The tensor caster accepted a spoofed Python type")
    result = probe.identity(value)
    assert result.data_ptr() == value.data_ptr()
    torch.testing.assert_close(result, value)
    assert probe.undefined() is None
    for method, args in ((torch.ops.nelux_abi._capture, (value,)),
                         (torch.ops.nelux_abi._export, ())):
        try:
            method(*args)
        except RuntimeError as error:
            assert "active conversion" in str(error)
        else:
            raise AssertionError("An exchange operator ran outside a conversion")

    class HostileTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, tensor):
            return torch.Tensor._make_subclass(cls, tensor)

        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            raise AssertionError("Tensor conversion invoked a subclass function override")

        @classmethod
        def __torch_dispatch__(cls, function, types, args=(), kwargs=None):
            raise AssertionError("Tensor conversion invoked a subclass dispatch override")

    hostile = HostileTensor(value)
    probe.identity(hostile)
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.overrides import TorchFunctionMode

    class HostileMode(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            raise AssertionError("Tensor conversion invoked a dispatch mode")

    with HostileMode():
        probe.identity(value)
        # Conversion must restore the existing user mode after it returns.
        try:
            value.clone()
        except AssertionError:
            pass
        else:
            raise AssertionError("Conversion failed to restore the dispatch mode")

    class HostileFunctionMode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            raise AssertionError("Tensor conversion invoked a function mode")

    with HostileFunctionMode():
        probe.identity(value)
        try:
            value.clone()
        except AssertionError:
            pass
        else:
            raise AssertionError("Conversion failed to restore the function mode")

    requires_grad = value.clone().requires_grad_()
    assert probe.identity(requires_grad).requires_grad
    with torch.inference_mode():
        inferred = value.clone()
        assert probe.identity(inferred).is_inference()

    # The cache must survive module eviction and retained native callables.
    import importlib
    retained_identity = probe.identity
    sys.modules.pop("abi_probe")
    reimported = importlib.import_module("abi_probe")
    assert retained_identity(value).data_ptr() == value.data_ptr()
    assert reimported.identity(value).data_ptr() == value.data_ptr()

    packet = torch.ops.nelux_abi._capture
    original = packet._op

    def failing_capture(tensor):
        raise RuntimeError("Intentional conversion failure")

    packet._op = failing_capture
    try:
        try:
            probe.identity(value)
        except RuntimeError as error:
            assert "Intentional conversion failure" in str(error)
        else:
            raise AssertionError("Expected a failed exchange")
    finally:
        packet._op = original
    assert probe.identity(value).data_ptr() == value.data_ptr()

    def nested_capture(tensor):
        assert probe.identity(tensor).data_ptr() == tensor.data_ptr()
        original(tensor)

    # Nest exactly once, then delegate to the original operator. The nested
    # conversion must restore the outer conversion's private context.
    def nesting_capture(tensor):
        packet._op = original
        try:
            nested_capture(tensor)
        finally:
            packet._op = nesting_capture

    packet._op = nesting_capture
    try:
        assert probe.identity(value).data_ptr() == value.data_ptr()
    finally:
        packet._op = original

    assert probe.stack(value).shape == (2, 2, 3)
    assert probe.permute(value).shape == (3, 2)
    assert probe.arange(4).tolist() == [0, 1, 2, 3]


def check_ownership():
    before = probe.deleted()
    value = probe.owned()
    assert value.tolist() == [[1, 2, 3], [4, 5, 6]]
    view = value[0]
    del value
    gc.collect()
    assert probe.deleted() == before
    del view
    gc.collect()
    assert probe.deleted() == before + 1
    assert probe.bad_blob() == before + 2


def check_arithmetic(device="cpu"):
    count = 0
    original_default = torch.get_default_dtype()
    try:
        for default in (torch.float32, torch.float64):
            torch.set_default_dtype(default)
            for dtype in (torch.bool, torch.uint8, torch.int8, torch.int16,
                          torch.int32, torch.int64, torch.float16,
                          torch.bfloat16, torch.float32, torch.float64):
                for scalar_input in (False, True):
                    values = 1 if scalar_input else [0, 1, 2, 3, 10, 127, 255, 65535]
                    value = torch.tensor(values, device=device).to(dtype)
                    for operation, scalar in (("mul_i", 255), ("mul_i", 257),
                                              ("mul_i", 65535), ("div_i", 65535),
                                              ("mul_f", 65535.0), ("mul_f", 255.0),
                                              ("div_f", 257.0)):
                        expected = value * scalar if operation.startswith("mul") else value / scalar
                        actual = getattr(probe, operation)(value, scalar)
                        assert actual.dtype == expected.dtype
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
                        count += 1
                    for minimum, maximum in ((0, 255), (0, 65535), (-1, 255)):
                        try:
                            expected = value.clamp(minimum, maximum)
                        except Exception:
                            try:
                                probe.clamp(value, minimum, maximum)
                            except Exception:
                                pass
                            else:
                                raise AssertionError(("Expected clamp rejection", dtype, minimum, maximum))
                        else:
                            actual = probe.clamp(value, minimum, maximum)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finally:
        torch.set_default_dtype(original_default)
    return count


if __name__ == "__main__":
    check_exchange()
    check_ownership()
    cases = check_arithmetic()
    print(f"PASS {torch.__version__}: {cases} arithmetic cases plus exchange and ownership")
