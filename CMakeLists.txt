cmake_minimum_required(VERSION 3.23)

# ─── Auto-detect vcpkg from local directory ───
# If CMAKE_TOOLCHAIN_FILE is not set, try to find vcpkg in the source directory
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND NOT DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
  set(_local_vcpkg_toolchain "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
  if(EXISTS "${_local_vcpkg_toolchain}")
    set(CMAKE_TOOLCHAIN_FILE "${_local_vcpkg_toolchain}" CACHE FILEPATH "vcpkg toolchain file")
    message(STATUS "Auto-detected local vcpkg: ${CMAKE_TOOLCHAIN_FILE}")
    
    # Also set default triplet if not set
    if(NOT DEFINED VCPKG_TARGET_TRIPLET AND NOT DEFINED ENV{VCPKG_TARGET_TRIPLET})
      if(WIN32)
        set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet")
      elseif(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
          set(VCPKG_TARGET_TRIPLET "arm64-osx-dynamic" CACHE STRING "vcpkg target triplet")
        else()
          set(VCPKG_TARGET_TRIPLET "x64-osx-dynamic" CACHE STRING "vcpkg target triplet")
        endif()
      else()
        set(VCPKG_TARGET_TRIPLET "x64-linux-dynamic" CACHE STRING "vcpkg target triplet")
      endif()
      message(STATUS "Using default triplet: ${VCPKG_TARGET_TRIPLET}")
    endif()
  endif()
endif()

project(Nelux VERSION 0.10.1 LANGUAGES CXX)

# ─── Compiler options ─── 
# Note: These apply to CXX compiler only. CUDA compiler options are set separately.
if (MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:/utf-8>                   # treat all .cpp/.hpp as UTF-8 (kills C4828)
    $<$<COMPILE_LANGUAGE:CXX>:/wd4267>                  # disable C4267: conversion from 'size_t' to 'int'
    $<$<COMPILE_LANGUAGE:CXX>:/wd4828>                  # disable C4828: illegal character in source
    $<$<COMPILE_LANGUAGE:CXX>:/external:anglebrackets>  # mark <> includes as "external"
    $<$<COMPILE_LANGUAGE:CXX>:/external:W0>             # suppress all warnings in external headers
  )
endif()

# Force only Release config for multi-config generators
if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)
endif()
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# C++20 is required by the PyTorch 2.13 headers: c10/util/StringUtil.h uses
# designated initializers and c10/core/AutogradState.h uses default member
# initializers for bit-fields. GCC/Clang accept both as C++17 extensions, so
# only MSVC rejects them (C7555 / C7582), and C++17 additionally leaves
# HeaderOnlyArrayRef's operator== ambiguous there (C2666).
set(CMAKE_CXX_STANDARD        20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NELUX_ENABLE_CUDA   "Enable CUDA support in Nelux" OFF)
option(NELUX_BUILD_PYTHON "Build Python bindings"     ON)
option(NELUX_BUNDLE_FFMPEG_DLLS "Bundle FFmpeg runtime DLLs into Windows wheels" ON)

# Host-side performance options
# - AVX2: targets modern x86_64 CPUs; can be disabled for broader compatibility.
#   Auto-disabled on non-x86 architectures (e.g. Apple Silicon arm64).
#   Applied to both CXX and CUDA host compilation.
# - LTO: enables interprocedural optimization (IPO/LTO) in Release when supported.
#   Applied to every Nelux target (NeluxLib, nelux, NeluxCuda).
# - CUDA_FAST_MATH: passes --use_fast_math to nvcc; safe for the NV12<->RGB
#   color-conversion kernels we ship.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86|i386|i686")
  set(_NELUX_AVX2_DEFAULT ON)
else()
  set(_NELUX_AVX2_DEFAULT OFF)
endif()
option(NELUX_ENABLE_AVX2            "Enable AVX2/FMA for host code (x86 only)" ${_NELUX_AVX2_DEFAULT})
option(NELUX_ENABLE_LTO             "Enable IPO/LTO for Release builds"        ON)
option(NELUX_ENABLE_CUDA_FAST_MATH  "Pass --use_fast_math to nvcc"             ON)

if(NELUX_ENABLE_AVX2 AND NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x86|i386|i686"))
  message(WARNING "NELUX_ENABLE_AVX2 requested on non-x86 target (${CMAKE_SYSTEM_PROCESSOR}); forcing OFF")
  set(NELUX_ENABLE_AVX2 OFF CACHE BOOL "" FORCE)
endif()

# Resolve IPO support once; used by nelux_apply_perf_flags below.
set(NELUX_IPO_SUPPORTED FALSE)
if(NELUX_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT NELUX_IPO_SUPPORTED OUTPUT _nelux_ipo_error)
  if(NOT NELUX_IPO_SUPPORTED)
    message(WARNING "IPO/LTO requested but not supported: ${_nelux_ipo_error}")
  endif()
endif()

# Apply host performance flags (AVX2/FMA, LTO) to a target. Covers CXX sources
# and CUDA host-side compilation (via nvcc -Xcompiler). No-op for languages
# that aren't present in the target.
function(nelux_apply_perf_flags target)
  if(NELUX_ENABLE_AVX2)
    if(MSVC)
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/arch:AVX2>
      )
    else()
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
        $<$<COMPILE_LANGUAGE:CXX>:-mfma>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx2>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mfma>
      )
    endif()
  endif()

  if(NELUX_ENABLE_LTO AND NELUX_IPO_SUPPORTED)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
endfunction()

# --- Enable CUDA language if requested ---
if(NELUX_ENABLE_CUDA)
  # Try to help CMake find CUDA if not already set
  if(NOT CMAKE_CUDA_COMPILER)
    # Check CUDA_PATH environment variable (set by NVIDIA installer)
    if(DEFINED ENV{CUDA_PATH})
      set(_cuda_path "$ENV{CUDA_PATH}")
      file(TO_CMAKE_PATH "${_cuda_path}" _cuda_path)
      if(WIN32)
        set(_nvcc_candidate "${_cuda_path}/bin/nvcc.exe")
      else()
        set(_nvcc_candidate "${_cuda_path}/bin/nvcc")
      endif()
      if(EXISTS "${_nvcc_candidate}")
        set(CMAKE_CUDA_COMPILER "${_nvcc_candidate}" CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "Found nvcc via CUDA_PATH: ${CMAKE_CUDA_COMPILER}")
      endif()
    endif()

    # Also check CUDAToolkit_ROOT
    if(NOT CMAKE_CUDA_COMPILER AND DEFINED ENV{CUDAToolkit_ROOT})
      set(_cuda_root "$ENV{CUDAToolkit_ROOT}")
      file(TO_CMAKE_PATH "${_cuda_root}" _cuda_root)
      if(WIN32)
        set(_nvcc_candidate "${_cuda_root}/bin/nvcc.exe")
      else()
        set(_nvcc_candidate "${_cuda_root}/bin/nvcc")
      endif()
      if(EXISTS "${_nvcc_candidate}")
        set(CMAKE_CUDA_COMPILER "${_nvcc_candidate}" CACHE FILEPATH "CUDA compiler" FORCE)
        message(STATUS "Found nvcc via CUDAToolkit_ROOT: ${CMAKE_CUDA_COMPILER}")
      endif()
    endif()
  endif()

  # Newer MSVC toolsets (e.g. VS 2026) may not yet be officially supported by
  # the installed CUDA Toolkit. NVCC hard-errors during compiler identification
  # unless we opt in to the override.
  #
  # We only enable this automatically for MSVC versions newer than VS 2022
  # (MSVC 19.3x => MSVC_VERSION 1930-1939).
  if(MSVC AND MSVC_VERSION GREATER 1939)
    string(APPEND CMAKE_CUDA_FLAGS_INIT " -allow-unsupported-compiler")
    message(STATUS "MSVC ${MSVC_VERSION} detected; adding NVCC flag: -allow-unsupported-compiler")
  endif()

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA enabled: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
    # Set CUDA architectures for common GPUs (can be overridden)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90" CACHE STRING "CUDA architectures")
    endif()
  else()
    message(FATAL_ERROR 
      "NELUX_ENABLE_CUDA is ON but no CUDA compiler found!\n"
      "The build cannot continue without CUDA when explicitly requested.\n"
      "Ensure CUDA Toolkit is installed and CUDA_PATH is set correctly.\n"
      "If you want a CPU-only build, set -DNELUX_ENABLE_CUDA=OFF"
    )
  endif()
endif()

# --- Compiler cache (ccache) ---
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  if(NELUX_ENABLE_CUDA)
    set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  endif()
endif()

# --- Dependencies ---
find_package(spdlog CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

# Support both exported names (fmt changed the hyphen/underscore in some builds)
if(TARGET fmt::fmt-header-only)
  set(FMT_HO_TARGET fmt::fmt-header-only)
elseif(TARGET fmt::fmt-header_only)
  set(FMT_HO_TARGET fmt::fmt-header_only)
else()
  message(FATAL_ERROR "fmt header-only target not found. Install a package that exports fmt::fmt-header-only.")
endif()

# --- FFmpeg ---
# Uses pre-built FFmpeg from external/ffmpeg directory
# Run: .\tools\download_ffmpeg.ps1 to download FFmpeg before building
set(NELUX_FFMPEG_DIR "${CMAKE_SOURCE_DIR}/external/ffmpeg")

if(NOT EXISTS "${NELUX_FFMPEG_DIR}/include/libavcodec/avcodec.h")
  message(FATAL_ERROR 
    "FFmpeg not found at ${NELUX_FFMPEG_DIR}\n"
    "Run: .\\tools\\download_ffmpeg.ps1 to download pre-built FFmpeg")
endif()

message(STATUS "Using external FFmpeg from: ${NELUX_FFMPEG_DIR}")

# Set include directories
set(NELUX_FFMPEG_INCS "${NELUX_FFMPEG_DIR}/include")

# Find import libraries
set(_ffmpeg_lib_dir "${NELUX_FFMPEG_DIR}/lib")
set(_ffmpeg_components avcodec avformat avutil avfilter swscale swresample avdevice)

if(WIN32)
  set(NELUX_FFMPEG_LIBS)
  foreach(_c ${_ffmpeg_components})
    if(MSVC)
      file(GLOB _defs "${_ffmpeg_lib_dir}/${_c}-*.def")
      if(NOT _defs)
        message(FATAL_ERROR "FFmpeg definition file not found for ${_c} in ${_ffmpeg_lib_dir}")
      endif()
      list(SORT _defs)
      list(REVERSE _defs)
      list(GET _defs 0 _def)

      set(_import_dir "${CMAKE_BINARY_DIR}/ffmpeg-msvc-import")
      file(MAKE_DIRECTORY "${_import_dir}")
      set(_import_lib "${_import_dir}/${_c}.lib")
      if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_machine x64)
      else()
        set(_machine x86)
      endif()
      execute_process(
        COMMAND "${CMAKE_AR}" /nologo /machine:${_machine} /def:${_def} /out:${_import_lib}
        RESULT_VARIABLE _lib_result
      )
      if(NOT _lib_result EQUAL 0)
        message(FATAL_ERROR "Failed to create MSVC FFmpeg import lib from ${_def}")
      endif()
      list(APPEND NELUX_FFMPEG_LIBS "${_import_lib}")
    else()
      list(APPEND NELUX_FFMPEG_LIBS "${_ffmpeg_lib_dir}/${_c}.lib")
    endif()
  endforeach()
elseif(APPLE)
  # Prefer unversioned symlink, fall back to highest-version .dylib found
  set(NELUX_FFMPEG_LIBS)
  foreach(_c ${_ffmpeg_components})
    set(_plain "${_ffmpeg_lib_dir}/lib${_c}.dylib")
    if(EXISTS "${_plain}")
      list(APPEND NELUX_FFMPEG_LIBS "${_plain}")
    else()
      file(GLOB _versioned "${_ffmpeg_lib_dir}/lib${_c}.*.dylib")
      if(_versioned)
        list(SORT _versioned)
        list(REVERSE _versioned)
        list(GET _versioned 0 _picked)
        list(APPEND NELUX_FFMPEG_LIBS "${_picked}")
      else()
        list(APPEND NELUX_FFMPEG_LIBS "${_plain}")
      endif()
    endif()
  endforeach()
else()
  # Linux: prefer unversioned .so symlink
  set(NELUX_FFMPEG_LIBS)
  foreach(_c ${_ffmpeg_components})
    set(_plain "${_ffmpeg_lib_dir}/lib${_c}.so")
    if(EXISTS "${_plain}")
      list(APPEND NELUX_FFMPEG_LIBS "${_plain}")
    else()
      file(GLOB _versioned "${_ffmpeg_lib_dir}/lib${_c}.so.*")
      if(_versioned)
        list(SORT _versioned)
        list(REVERSE _versioned)
        list(GET _versioned 0 _picked)
        list(APPEND NELUX_FFMPEG_LIBS "${_picked}")
      else()
        list(APPEND NELUX_FFMPEG_LIBS "${_plain}")
      endif()
    endif()
  endforeach()
endif()

# Verify libraries exist
foreach(_lib ${NELUX_FFMPEG_LIBS})
  if(NOT EXISTS "${_lib}")
    message(FATAL_ERROR "FFmpeg library not found: ${_lib}")
  endif()
endforeach()

message(STATUS "FFmpeg includes: ${NELUX_FFMPEG_INCS}")
message(STATUS "FFmpeg libraries: ${NELUX_FFMPEG_LIBS}")

if(NELUX_BUILD_PYTHON)
  find_package(Python3 3.13 REQUIRED COMPONENTS Interpreter Development.Module)
  message(STATUS "Python3: ${Python3_EXECUTABLE}")
  message(STATUS "Python3 includes: ${Python3_INCLUDE_DIRS}")

  # Determine site-packages path (needed for PyTorch and fallback pybind11)
  execute_process(
    COMMAND ${Python3_EXECUTABLE}
            -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE Python3_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  message(STATUS "Python site-packages prefix: ${Python3_SITE_PACKAGES}")

  # -- Find pybind11 --
  find_package(pybind11 CONFIG)
  if(pybind11_FOUND)
    message(STATUS "pybind11 (CMake): ${pybind11_VERSION}")
    set(NELUX_PYBIND11_INCS ${pybind11_INCLUDE_DIRS})
    set(NELUX_PYBIND11_LIBS pybind11::module)
  else()
    message(STATUS "pybind11 not found via CMake; falling back to pip headers")
    
    if(WIN32)
      set(NELUX_PYBIND11_INCS
          "${Python3_SITE_PACKAGES}/Lib/site-packages/pybind11/include")
    else()
      set(NELUX_PYBIND11_INCS
          "${Python3_SITE_PACKAGES}/pybind11/include")
    endif()
    set(NELUX_PYBIND11_LIBS "")
    if(NOT EXISTS "${NELUX_PYBIND11_INCS}/pybind11/pybind11.h")
      message(FATAL_ERROR "pybind11 headers missing; run: pip install pybind11")
    endif()
  endif()

  # -- PyTorch paths (detected dynamically via Python import) --
  execute_process(
    COMMAND ${Python3_EXECUTABLE}
            -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_PACKAGE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_result
  )
  
  if(_torch_result EQUAL 0 AND EXISTS "${TORCH_PACKAGE_DIR}")
    set(NELUX_PYTORCH_LIBDIR "${TORCH_PACKAGE_DIR}/lib")
    set(NELUX_PYTORCH_INCLUDEDIR "${TORCH_PACKAGE_DIR}/include")
    execute_process(
      COMMAND ${Python3_EXECUTABLE}
              -c "import torch; v=torch.__version__.split('+', 1)[0].split('.')[:2]; print('.'.join(v))"
      OUTPUT_VARIABLE NELUX_TORCH_ABI
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    # Main headers: torch/include and torch/include/torch/csrc/api/include
    include_directories(
      ${NELUX_PYTORCH_INCLUDEDIR}
      ${NELUX_PYTORCH_INCLUDEDIR}/torch/csrc/api/include
    )
    # Add the lib directory to the linker
    link_directories(${NELUX_PYTORCH_LIBDIR})
    
    message(STATUS "Using torch headers: ${NELUX_PYTORCH_INCLUDEDIR}")
    message(STATUS "Using torch libs:    ${NELUX_PYTORCH_LIBDIR}")
    message(STATUS "Building Nelux against PyTorch ABI: ${NELUX_TORCH_ABI}")
  else()
    message(FATAL_ERROR "PyTorch not found. Run: pip install torch")
  endif()
endif()

file(GLOB_RECURSE NELUX_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Nelux/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Nelux/conversion/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Nelux/backends/*.cpp
)
list(REMOVE_ITEM NELUX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Nelux/FFmpegDelayLoad.cpp)

# Add CUDA sources if CUDA is enabled  
if(NELUX_ENABLE_CUDA)
  # Add only the CUDA cpp files to main sources, but NOT the .cu files
  file(GLOB NELUX_CUDA_CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Nelux/backends/cuda/*.cpp
  )
  list(APPEND NELUX_SOURCES ${NELUX_CUDA_CPP_SOURCES})
  
  # Create a separate CUDA library for .cu files
  file(GLOB NELUX_CUDA_CU_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Nelux/backends/cuda/*.cu
  )
  
  if(NELUX_CUDA_CU_SOURCES)
    add_library(NeluxCuda STATIC ${NELUX_CUDA_CU_SOURCES})
    # Static-link the CUDA runtime on every platform so cudart is not an eager
    # dependency of _nelux (libcudart.so / cudart64_*.dll). CPU-only torch ships
    # no CUDA runtime and the GPU paths that use it are runtime-gated, so the
    # module must load without it. Windows was already de-facto static (no cudart
    # in the import table); making it explicit closes the gap if a future
    # toolchain default changes.
    set_target_properties(NeluxCuda PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      POSITION_INDEPENDENT_CODE ON
      CUDA_STANDARD 20
      CUDA_RUNTIME_LIBRARY Static
    )
    target_include_directories(NeluxCuda PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux
      ${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/backends
      ${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/backends/cuda
      ${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/conversion
      ${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/conversion/gpu
    )
    target_compile_definitions(NeluxCuda PUBLIC NELUX_ENABLE_CUDA)

    nelux_apply_perf_flags(NeluxCuda)

    if(NELUX_ENABLE_CUDA_FAST_MATH)
      target_compile_options(NeluxCuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
      )
    endif()

    # Do NOT link CUDA::cudart here - let the runtime be resolved at link time
    message(STATUS "CUDA .cu sources: ${NELUX_CUDA_CU_SOURCES}")
  endif()
  
  message(STATUS "CUDA .cpp sources: ${NELUX_CUDA_CPP_SOURCES}")
endif()

add_library(NeluxLib STATIC ${NELUX_SOURCES})

nelux_apply_perf_flags(NeluxLib)

# Link CUDA library if enabled
if(NELUX_ENABLE_CUDA)
  target_compile_definitions(NeluxLib PUBLIC NELUX_ENABLE_CUDA)
  if(TARGET NeluxCuda)
    target_link_libraries(NeluxLib PUBLIC NeluxCuda)
  endif()
  # Static cudart so _nelux does not list a CUDA runtime as an eager dependency
  # (libcudart.so / cudart64_*.dll — absent on CPU-only torch). Covers cudart
  # calls from CXX TUs such as the NVDEC Decoder; pairs with NeluxCuda's
  # CUDA_RUNTIME_LIBRARY=Static. (cudart_static pulls -ldl/-lrt/-lpthread on
  # Linux via CMake.)
  if(TARGET CUDA::cudart_static)
    target_link_libraries(NeluxLib PUBLIC CUDA::cudart_static)
  endif()
endif()

target_include_directories(NeluxLib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/backends>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/backends/cuda>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/python>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/conversion>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/conversion/cpu>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/conversion/gpu>
    $<INSTALL_INTERFACE:include>
    ${NELUX_FFMPEG_INCS}
    $<$<BOOL:${NELUX_BUILD_PYTHON}>:${Python3_INCLUDE_DIRS}>
    $<$<BOOL:${NELUX_ENABLE_CUDA}>:${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}>
)

target_link_libraries(NeluxLib PUBLIC
  ${NELUX_FFMPEG_LIBS}
  spdlog::spdlog_header_only
  ${FMT_HO_TARGET}
  $<$<BOOL:${NELUX_BUILD_PYTHON}>:torch_cpu>
  $<$<BOOL:${NELUX_BUILD_PYTHON}>:c10>
  $<$<BOOL:${NELUX_BUILD_PYTHON}>:torch>
  $<$<BOOL:${NELUX_BUILD_PYTHON}>:torch_python>
  # Windows ONLY: c10_cuda/torch_cuda are delay-loaded (see /DELAYLOAD) so they
  # stay out of the eager import table while the import lib is present for the
  # delay-load thunk. On Linux a plain -lc10_cuda becomes a DT_NEEDED even when
  # unused (no --as-needed), which would break import on CPU-only torch; there
  # the symbol is resolved at runtime via dlsym (src/Nelux/CudaStream.cpp).
  $<$<AND:$<BOOL:${NELUX_BUILD_PYTHON}>,$<BOOL:${NELUX_ENABLE_CUDA}>,$<PLATFORM_ID:Windows>>:torch_cuda>
  $<$<AND:$<BOOL:${NELUX_BUILD_PYTHON}>,$<BOOL:${NELUX_ENABLE_CUDA}>,$<PLATFORM_ID:Windows>>:c10_cuda>
)
if(WIN32)
  target_link_libraries(NeluxLib PUBLIC ws2_32 bcrypt)
elseif(APPLE)
  target_link_libraries(NeluxLib PUBLIC pthread dl)
elseif(UNIX)
  target_link_libraries(NeluxLib PUBLIC pthread dl rt)
endif()

if(NELUX_BUILD_PYTHON)
  set(NELUX_PYTHON_SOURCES src/Nelux/python/Bindings.cpp)
  if(WIN32)
    list(APPEND NELUX_PYTHON_SOURCES src/Nelux/FFmpegDelayLoad.cpp)
  endif()
  add_library(nelux MODULE ${NELUX_PYTHON_SOURCES})

  nelux_apply_perf_flags(nelux)

  target_include_directories(nelux PRIVATE
    ${NELUX_PYBIND11_INCS}
    ${Python3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Nelux/python
  )

  target_link_libraries(nelux PRIVATE
    NeluxLib
    Python3::Module
    ${NELUX_PYBIND11_LIBS}
    $<$<BOOL:${NELUX_BUILD_PYTHON}>:torch_cpu>
    $<$<BOOL:${NELUX_BUILD_PYTHON}>:c10>
    $<$<BOOL:${NELUX_BUILD_PYTHON}>:torch>
    $<$<BOOL:${NELUX_BUILD_PYTHON}>:torch_python>
    # Windows ONLY (mirrors NeluxLib above): linked so the /DELAYLOAD thunk has
    # its import lib. On Linux a plain -lc10_cuda/-ltorch_cuda becomes a
    # DT_NEEDED even when unused and breaks import on CPU-only torch; the symbol
    # is resolved at runtime via dlsym (src/Nelux/CudaStream.cpp).
    $<$<AND:$<BOOL:${NELUX_BUILD_PYTHON}>,$<BOOL:${NELUX_ENABLE_CUDA}>,$<PLATFORM_ID:Windows>>:torch_cuda>
    $<$<AND:$<BOOL:${NELUX_BUILD_PYTHON}>,$<BOOL:${NELUX_ENABLE_CUDA}>,$<PLATFORM_ID:Windows>>:c10_cuda>
  )

  # Delay-load FFmpeg DLLs on Windows for version flexibility
  if(WIN32 AND MSVC)
    # Link delayimp.lib for delay-load support
    target_link_libraries(nelux PRIVATE delayimp)
    
    file(GLOB _nelux_ffmpeg_delay_dlls "${NELUX_FFMPEG_DIR}/bin/av*.dll" "${NELUX_FFMPEG_DIR}/bin/sw*.dll")
    foreach(_dll ${_nelux_ffmpeg_delay_dlls})
      get_filename_component(_dll_name "${_dll}" NAME)
      if(_dll_name MATCHES "^(avcodec|avformat|avutil|swscale|swresample|avfilter|avdevice)-.*\\.dll$")
        target_link_options(nelux PRIVATE "/DELAYLOAD:${_dll_name}")
      endif()
    endforeach()
    message(STATUS "MSVC: FFmpeg DLLs will be delay-loaded for version flexibility")
  endif()

  if(NELUX_ENABLE_CUDA)
    target_compile_definitions(nelux PRIVATE NELUX_ENABLE_CUDA)

    if(MSVC AND TARGET NeluxCuda)
      target_link_options(nelux PRIVATE
        "/WHOLEARCHIVE:$<TARGET_FILE:NeluxCuda>"
      )
      message(STATUS "MSVC: Using /WHOLEARCHIVE for NeluxCuda to include all CUDA kernels in nelux")
    endif()

    if(WIN32 AND MSVC)
      # Delay-load the CUDA PyTorch runtime so the module still IMPORTS on a
      # CPU-only PyTorch (where c10_cuda.dll is absent). Every c10::cuda / CUDA
      # code path is runtime-gated (frame.is_cuda(), NVDEC accelerator), so
      # c10_cuda is only resolved when a GPU op actually runs. cudart is
      # static-linked and the NVDEC driver DLLs are LoadLibrary'd at runtime, so
      # neither appears in the import table. Without this, the eager c10_cuda
      # import makes `import nelux` fail with WinError 126 on CPU-only torch.
      # (torch_cuda.dll is listed defensively; if a build doesn't import it the
      #  linker emits a benign LNK4199 and ignores the directive.)
      # delayimp.lib is already linked above (FFmpeg delay-load block).
      target_link_options(nelux PRIVATE
        "/DELAYLOAD:c10_cuda.dll"
        "/DELAYLOAD:torch_cuda.dll"
      )
      message(STATUS "MSVC: c10_cuda.dll/torch_cuda.dll delay-loaded — import works on CPU-only PyTorch")
    endif()
  endif()

  if(NELUX_TORCH_ABI)
    target_compile_definitions(nelux PRIVATE NELUX_TORCH_ABI="${NELUX_TORCH_ABI}")
  endif()

  set_target_properties(nelux PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_nelux"

    # single-config fallback
    LIBRARY_OUTPUT_DIRECTORY        ${CMAKE_CURRENT_SOURCE_DIR}/nelux
    RUNTIME_OUTPUT_DIRECTORY        ${CMAKE_CURRENT_SOURCE_DIR}/nelux

    # multi-config overrides:
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/nelux
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/nelux
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/nelux
    RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${CMAKE_CURRENT_SOURCE_DIR}/nelux
  )
  if(WIN32)
    set_target_properties(nelux PROPERTIES SUFFIX ".pyd")
  else()
    set_target_properties(nelux PROPERTIES SUFFIX ".so")
  endif()

  # RPATH so nelux/_nelux.so|.dylib finds sibling deps at runtime.
  if(APPLE)
    set_target_properties(nelux PROPERTIES
      INSTALL_RPATH "@loader_path"
      BUILD_WITH_INSTALL_RPATH TRUE
      MACOSX_RPATH TRUE
    )
  elseif(UNIX)
    set_target_properties(nelux PROPERTIES
      INSTALL_RPATH "$ORIGIN"
      BUILD_WITH_INSTALL_RPATH TRUE
    )
  endif()
endif()

if(WIN32 AND NELUX_BUILD_PYTHON)
  # DLL copy logic for Windows only
  # Bundle runtime DLLs into the wheel so self-built wheels import cleanly.
  set(NELUX_DLL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nelux")
  
  # vcpkg provides spdlog, fmt
  if(DEFINED ENV{VCPKG_TARGET_TRIPLET})
    set(_vcpkg_triplet "$ENV{VCPKG_TARGET_TRIPLET}")
  else()
    set(_vcpkg_triplet "x64-windows")
  endif()
  
  # vcpkg can put DLLs in different locations depending on mode:
  # - Classic mode: vcpkg/installed/<triplet>/bin/
  # - Manifest mode: vcpkg/packages/<package>_<triplet>/bin/
  if(DEFINED ENV{VCPKG_ROOT})
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _vcpkg_root_cmake)
    set(VCPKG_INSTALLED_BIN "${_vcpkg_root_cmake}/installed/${_vcpkg_triplet}/bin")
    set(VCPKG_PACKAGES_DIR "${_vcpkg_root_cmake}/packages")
  else()
    set(VCPKG_INSTALLED_BIN "${CMAKE_SOURCE_DIR}/vcpkg/installed/${_vcpkg_triplet}/bin")
    set(VCPKG_PACKAGES_DIR "${CMAKE_SOURCE_DIR}/vcpkg/packages")
  endif()
  
  # Try installed directory first (classic mode)
  file(GLOB NELUX_DLLS
    "${VCPKG_INSTALLED_BIN}/spdlog*.dll"
    "${VCPKG_INSTALLED_BIN}/fmt*.dll"
    "${VCPKG_INSTALLED_BIN}/jpeg*.dll"
    "${VCPKG_INSTALLED_BIN}/turbojpeg*.dll"
  )

  # Also try manifest-mode vcpkg_installed locations in the current build tree
  # (common with scikit-build temp build directories).
  file(GLOB NELUX_DLLS_BUILD_TREE
    "${CMAKE_BINARY_DIR}/vcpkg_installed/*/bin/spdlog*.dll"
    "${CMAKE_BINARY_DIR}/vcpkg_installed/*/bin/fmt*.dll"
    "${CMAKE_BINARY_DIR}/vcpkg_installed/*/bin/jpeg*.dll"
    "${CMAKE_BINARY_DIR}/vcpkg_installed/*/bin/turbojpeg*.dll"
  )
  if(NELUX_DLLS_BUILD_TREE)
    list(APPEND NELUX_DLLS ${NELUX_DLLS_BUILD_TREE})
  endif()

  # If not found, try packages directory (manifest mode)
  if(NOT NELUX_DLLS)
    message(STATUS "DLLs not found in installed directory, checking packages directory...")
    file(GLOB NELUX_DLLS
      "${VCPKG_PACKAGES_DIR}/spdlog_${_vcpkg_triplet}/bin/spdlog*.dll"
      "${VCPKG_PACKAGES_DIR}/fmt_${_vcpkg_triplet}/bin/fmt*.dll"
      "${VCPKG_PACKAGES_DIR}/libjpeg-turbo_${_vcpkg_triplet}/bin/jpeg*.dll"
      "${VCPKG_PACKAGES_DIR}/libjpeg-turbo_${_vcpkg_triplet}/bin/turbojpeg*.dll"
    )
  endif()

  # CUDA runtime DLLs (cudart, nvrtc, nvrtc-builtins, …) are intentionally NOT
  # bundled. PyTorch already ships them in torch/lib/ and loads them before
  # nelux is imported (nelux/__init__.py enforces `import torch` first), so
  # bundling our own copies would add ~100 MB of dead weight that duplicates
  # what the user's PyTorch install already provides.

  if(NELUX_BUNDLE_FFMPEG_DLLS)
    # Bundle FFmpeg runtime DLLs from external/ffmpeg/bin when explicitly enabled.
    file(GLOB NELUX_FFMPEG_DLLS
      "${NELUX_FFMPEG_DIR}/bin/avcodec-*.dll"
      "${NELUX_FFMPEG_DIR}/bin/avformat-*.dll"
      "${NELUX_FFMPEG_DIR}/bin/avutil-*.dll"
      "${NELUX_FFMPEG_DIR}/bin/swscale-*.dll"
      "${NELUX_FFMPEG_DIR}/bin/swresample-*.dll"
      "${NELUX_FFMPEG_DIR}/bin/avfilter-*.dll"
      "${NELUX_FFMPEG_DIR}/bin/avdevice-*.dll"
    )
    if(NELUX_FFMPEG_DLLS)
      list(APPEND NELUX_DLLS ${NELUX_FFMPEG_DLLS})
    else()
      message(WARNING "FFmpeg runtime DLLs not found under ${NELUX_FFMPEG_DIR}/bin")
    endif()
  else()
    message(STATUS "NELUX_BUNDLE_FFMPEG_DLLS=OFF, skipping FFmpeg runtime DLL bundling")
  endif()

  # Deduplicate in case multiple glob sources overlap
  if(NELUX_DLLS)
    list(REMOVE_DUPLICATES NELUX_DLLS)
  endif()

  message(STATUS "Found ${CMAKE_CURRENT_LIST_DIR} DLLs for delvewheel: ${NELUX_DLLS}")

  # Keep local dev copy option for in-tree testing.
  if(DEFINED ENV{NELUX_COPY_DLLS_FOR_DEV})
    foreach(dll ${NELUX_DLLS})
      add_custom_command(TARGET nelux POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${dll}" "${NELUX_DLL_DIR}"
        COMMENT "Copying dependency DLL for dev: ${dll} → ${NELUX_DLL_DIR}"
      )
    endforeach()
  endif()


endif()

if(UNIX AND NELUX_BUILD_PYTHON)
  # Locate the vcpkg install tree for this triplet. NOTE: FFmpeg libs are NOT
  # bundled here — users must have FFmpeg installed (or use auditwheel/delocate
  # at wheel-repair time).
  if(DEFINED ENV{VCPKG_TARGET_TRIPLET})
    set(_nelux_triplet "$ENV{VCPKG_TARGET_TRIPLET}")
  elseif(DEFINED VCPKG_TARGET_TRIPLET)
    set(_nelux_triplet "${VCPKG_TARGET_TRIPLET}")
  elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(_nelux_triplet "arm64-osx-dynamic")
    else()
      set(_nelux_triplet "x64-osx-dynamic")
    endif()
  else()
    set(_nelux_triplet "x64-linux-dynamic")
  endif()

  if(DEFINED ENV{VCPKG_ROOT})
    file(TO_CMAKE_PATH "$ENV{VCPKG_ROOT}" _vcpkg_root_cmake)
    set(VCPKG_LIB "${_vcpkg_root_cmake}/installed/${_nelux_triplet}/lib")
  else()
    set(VCPKG_LIB "${CMAKE_SOURCE_DIR}/vcpkg/installed/${_nelux_triplet}/lib")
  endif()

  set(NELUX_SO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nelux")

  if(APPLE)
    file(GLOB NELUX_SOS
      "${VCPKG_LIB}/libspdlog*.dylib"
      "${VCPKG_LIB}/libfmt*.dylib"
    )
  else()
    file(GLOB NELUX_SOS
      "${VCPKG_LIB}/libspdlog.so*"
      "${VCPKG_LIB}/libfmt.so*"
    )
  endif()

  foreach(sofile ${NELUX_SOS})
    add_custom_command(TARGET nelux POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${sofile}"
        "${NELUX_SO_DIR}"
      COMMENT "Copying dependency shared lib: ${sofile} -> ${NELUX_SO_DIR}"
    )
  endforeach()
endif()

install(TARGETS nelux
  LIBRARY   DESTINATION nelux   # for .so on Linux
  RUNTIME   DESTINATION nelux   # for .pyd on Windows
)

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/nelux/__init__.py
  ${CMAKE_CURRENT_SOURCE_DIR}/nelux/batch.py
  ${CMAKE_CURRENT_SOURCE_DIR}/nelux/py.typed
  ${CMAKE_CURRENT_SOURCE_DIR}/nelux/_nelux.pyi
  DESTINATION nelux
)

if(WIN32 AND NELUX_BUILD_PYTHON)
  # In CI, delvewheel handles vcpkg DLL bundling via --add-path. Installing them
  # into nelux/ here would duplicate every DLL (one in nelux/, one hashed copy
  # in nelux.libs/). Set NELUX_SKIP_DLL_INSTALL=ON in CI to opt out; default
  # remains ON so local dev (`pip install -e .`) gets DLLs next to _nelux.pyd.
  option(NELUX_SKIP_DLL_INSTALL "Skip installing vcpkg DLLs into nelux/ (use when delvewheel will repair the wheel)" OFF)
  if(DEFINED ENV{NELUX_SKIP_DLL_INSTALL})
    set(NELUX_SKIP_DLL_INSTALL "$ENV{NELUX_SKIP_DLL_INSTALL}")
  endif()

  if(NELUX_SKIP_DLL_INSTALL)
    message(STATUS "NELUX_SKIP_DLL_INSTALL=ON — leaving vcpkg DLL bundling to delvewheel")
  elseif(NELUX_DLLS)
    install(FILES ${NELUX_DLLS} DESTINATION nelux)
    message(STATUS "Bundling ${CMAKE_CURRENT_LIST_DIR} runtime DLLs into wheel: ${NELUX_DLLS}")
  else()
    message(WARNING "No runtime DLLs found to bundle into wheel on Windows")
  endif()
endif()

# debug dump of the variables we care about
message(STATUS "---- CMake Debug Dump ----")
message(STATUS "  CMAKE_TOOLCHAIN_FILE = ${CMAKE_TOOLCHAIN_FILE}")
message(STATUS "  VCPKG_ROOT (env)       = $ENV{VCPKG_ROOT}")
message(STATUS "  VCPKG_TARGET_TRIPLET   = $ENV{VCPKG_TARGET_TRIPLET}")
message(STATUS "  CMAKE_PREFIX_PATH      = ${CMAKE_PREFIX_PATH}")
message(STATUS "  CMAKE_MODULE_PATH      = ${CMAKE_MODULE_PATH}")
message(STATUS "  Python3_EXECUTABLE     = ${Python3_EXECUTABLE}")
message(STATUS "  Python3_ROOT_DIR       = ${Python3_ROOT_DIR}")
message(STATUS "----------------------------")
