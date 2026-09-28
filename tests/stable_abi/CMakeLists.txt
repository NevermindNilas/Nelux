cmake_minimum_required(VERSION 3.20)
project(nelux_stable_boundary_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import pathlib,torch; print(pathlib.Path(torch.__file__).parent)"
    OUTPUT_VARIABLE PROBE_TORCH_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
get_filename_component(PROBE_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
find_library(PROBE_TORCH_CPU NAMES torch_cpu PATHS "${PROBE_TORCH_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
pybind11_add_module(abi_probe probe.cpp
    "${PROBE_REPO_ROOT}/src/Nelux/TensorSupport.cpp"
    "${PROBE_REPO_ROOT}/src/Nelux/python/TensorInterop.cpp")
target_include_directories(abi_probe PRIVATE
    "${PROBE_REPO_ROOT}/include/Nelux" "${PROBE_REPO_ROOT}/include/Nelux/python")
target_include_directories(abi_probe BEFORE PRIVATE ${pybind11_INCLUDE_DIRS})
# Torch bundles a patched ordinary-ABI pybind11; external pybind11 must win.
target_include_directories(abi_probe SYSTEM PRIVATE "${PROBE_TORCH_ROOT}/include")
target_compile_definitions(abi_probe PRIVATE TORCH_TARGET_VERSION=0x020c000000000000)
target_link_libraries(abi_probe PRIVATE "${PROBE_TORCH_CPU}")

# Fault injection is confined to a standalone executable and never enters the
# production module. Successful paths still use real stable Tensor/storage.
add_executable(ownership_fault_probe ownership_fault_probe.cpp)
find_package(Threads REQUIRED)
target_include_directories(ownership_fault_probe PRIVATE "${PROBE_REPO_ROOT}/include/Nelux")
target_include_directories(ownership_fault_probe SYSTEM PRIVATE "${PROBE_TORCH_ROOT}/include")
target_compile_definitions(ownership_fault_probe PRIVATE TORCH_TARGET_VERSION=0x020c000000000000)
target_link_libraries(ownership_fault_probe PRIVATE "${PROBE_TORCH_CPU}" Threads::Threads)
set_target_properties(ownership_fault_probe PROPERTIES BUILD_RPATH "${PROBE_TORCH_ROOT}/lib")
