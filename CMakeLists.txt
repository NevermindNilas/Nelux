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

project(Nelux VERSION 0.9.2 LANGUAGES CXX)

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
set(CMAKE_CXX_STANDARD        17)
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
find_package(libyuv REQUIRED)

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
set(_ffmpeg_components avcodec avformat avutil avfilter swscale avdevice)

if(WIN32)
  set(NELUX_FFMPEG_LIBS)
  foreach(_c ${_ffmpeg_components})
    list(APPEND NELUX_FFMPEG_LIBS "${_ffmpeg_lib_dir}/${_c}.lib")
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
    
    # Main headers: torch/include and torch/include/torch/csrc/api/include
    include_directories(
      ${NELUX_PYTORCH_INCLUDEDIR}
      ${NELUX_PYTORCH_INCLUDEDIR}/torch/csrc/api/include
    )
    # Add the lib directory to the linker
    link_directories(${NELUX_PYTORCH_LIBDIR})
    
    message(STATUS "Using torch headers: ${NELUX_PYTORCH_INCLUDEDIR}")
    message(STATUS "Using torch libs:    ${NELUX_PYTORCH_LIBDIR}")
  else()
    message(FATAL_ERROR "PyTorch not found. Run: pip install torch")
  endif()
endif()

file(GLOB_RECURSE NELUX_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Nelux/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Nelux/conversion/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Nelux/backends/*.cpp
)

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
    set_target_properties(NeluxCuda PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      POSITION_INDEPENDENT_CODE ON
      CUDA_STANDARD 17
      CUDA_RUNTIME_LIBRARY Shared
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

# Add delay-load hook for FFmpeg on Windows
if(WIN32)
  list(APPEND NELUX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Nelux/FFmpegDelayLoad.cpp)
endif()

add_library(NeluxLib STATIC ${NELUX_SOURCES})

nelux_apply_perf_flags(NeluxLib)

# Link CUDA library if enabled
if(NELUX_ENABLE_CUDA)
  target_compile_definitions(NeluxLib PUBLIC NELUX_ENABLE_CUDA)
  if(TARGET NeluxCuda)
    target_link_libraries(NeluxLib PUBLIC NeluxCuda)
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
  yuv
  $<$<BOOL:${NELUX_BUILD_PYTHON}>:torch_cpu>
  $<$<BOOL:${NELUX_BUILD_PYTHON}>:c10>
  $<$<BOOL:${NELUX_BUILD_PYTHON}>:torch>
  $<$<BOOL:${NELUX_BUILD_PYTHON}>:torch_python>
  $<$<AND:$<BOOL:${NELUX_BUILD_PYTHON}>,$<BOOL:${NELUX_ENABLE_CUDA}>>:torch_cuda>
  $<$<AND:$<BOOL:${NELUX_BUILD_PYTHON}>,$<BOOL:${NELUX_ENABLE_CUDA}>>:c10_cuda>
)
if(WIN32)
  target_link_libraries(NeluxLib PUBLIC ws2_32 bcrypt)
elseif(APPLE)
  target_link_libraries(NeluxLib PUBLIC pthread dl)
elseif(UNIX)
  target_link_libraries(NeluxLib PUBLIC pthread dl rt)
endif()

if(NELUX_BUILD_PYTHON)
  add_library(nelux MODULE src/Nelux/python/Bindings.cpp)

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
    $<$<AND:$<BOOL:${NELUX_BUILD_PYTHON}>,$<BOOL:${NELUX_ENABLE_CUDA}>>:torch_cuda>
    $<$<AND:$<BOOL:${NELUX_BUILD_PYTHON}>,$<BOOL:${NELUX_ENABLE_CUDA}>>:c10_cuda>
  )

  # Delay-load FFmpeg DLLs on Windows for version flexibility
  if(WIN32 AND MSVC)
    # Link delayimp.lib for delay-load support
    target_link_libraries(nelux PRIVATE delayimp)
    
    # Delay-load FFmpeg DLLs - they will be resolved at runtime via our hook
    target_link_options(nelux PRIVATE
      "/DELAYLOAD:avcodec-62.dll"
      "/DELAYLOAD:avformat-62.dll"
      "/DELAYLOAD:avutil-60.dll"
      "/DELAYLOAD:swscale-9.dll"
      "/DELAYLOAD:avfilter-11.dll"
      "/DELAYLOAD:avdevice-62.dll"
    )
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
  
  # vcpkg provides spdlog, fmt, libyuv
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
    "${VCPKG_INSTALLED_BIN}/libyuv*.dll"
    "${VCPKG_INSTALLED_BIN}/spdlog*.dll"
    "${VCPKG_INSTALLED_BIN}/fmt*.dll"
    "${VCPKG_INSTALLED_BIN}/jpeg*.dll"
    "${VCPKG_INSTALLED_BIN}/turbojpeg*.dll"
  )

  # Also try manifest-mode vcpkg_installed locations in the current build tree
  # (common with scikit-build temp build directories).
  file(GLOB NELUX_DLLS_BUILD_TREE
    "${CMAKE_BINARY_DIR}/vcpkg_installed/*/bin/libyuv*.dll"
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
      "${VCPKG_PACKAGES_DIR}/libyuv_${_vcpkg_triplet}/bin/libyuv*.dll"
      "${VCPKG_PACKAGES_DIR}/spdlog_${_vcpkg_triplet}/bin/spdlog*.dll"
      "${VCPKG_PACKAGES_DIR}/fmt_${_vcpkg_triplet}/bin/fmt*.dll"
      "${VCPKG_PACKAGES_DIR}/libjpeg-turbo_${_vcpkg_triplet}/bin/jpeg*.dll"
      "${VCPKG_PACKAGES_DIR}/libjpeg-turbo_${_vcpkg_triplet}/bin/turbojpeg*.dll"
    )
  endif()

  if(NELUX_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "[CUDA DEBUG] CUDAToolkit_FOUND: ${CUDAToolkit_FOUND}")
    message(STATUS "[CUDA DEBUG] CUDAToolkit_BIN_DIR: ${CUDAToolkit_BIN_DIR}")
    message(STATUS "[CUDA DEBUG] ENV{CUDA_PATH}: $ENV{CUDA_PATH}")
    
    if(CUDAToolkit_FOUND AND DEFINED CUDAToolkit_BIN_DIR)
      file(TO_CMAKE_PATH "${CUDAToolkit_BIN_DIR}" CUDA_BIN_DIR)
    elseif(DEFINED ENV{CUDA_PATH})
      file(TO_CMAKE_PATH "$ENV{CUDA_PATH}/bin" CUDA_BIN_DIR)
    endif()

    message(STATUS "[CUDA DEBUG] Resolved CUDA_BIN_DIR: ${CUDA_BIN_DIR}")

    if(DEFINED CUDA_BIN_DIR)
      # List all DLLs in CUDA bin for debugging
      file(GLOB _cuda_bin_all_dlls "${CUDA_BIN_DIR}/*.dll")
      message(STATUS "[CUDA DEBUG] All DLLs in CUDA bin (first 10): ")
      list(LENGTH _cuda_bin_all_dlls _cuda_dll_count)
      if(_cuda_dll_count GREATER 0)
        list(SUBLIST _cuda_bin_all_dlls 0 10 _cuda_bin_sample)
        foreach(_dll ${_cuda_bin_sample})
          message(STATUS "  - ${_dll}")
        endforeach()
      else()
        message(WARNING "[CUDA DEBUG] NO DLLs found in ${CUDA_BIN_DIR}!")
      endif()

      # CUDA 13+ stores DLLs in bin/x64 subdirectory
      file(GLOB NELUX_CUDA_DLLS
        "${CUDA_BIN_DIR}/x64/cudart64_*.dll"
        "${CUDA_BIN_DIR}/x64/nvrtc64_*.dll"
        "${CUDA_BIN_DIR}/x64/nvrtc-builtins64_*.dll"
        # Also try root bin directory for older CUDA versions
        "${CUDA_BIN_DIR}/cudart64_*.dll"
        "${CUDA_BIN_DIR}/nvrtc64_*.dll"
        "${CUDA_BIN_DIR}/nvrtc-builtins64_*.dll"
      )
      
      list(LENGTH NELUX_CUDA_DLLS _cuda_dlls_found)
      if(_cuda_dlls_found GREATER 0)
        list(APPEND NELUX_DLLS ${NELUX_CUDA_DLLS})
        message(STATUS "[CUDA DEBUG] Found ${_cuda_dlls_found} CUDA DLLs to bundle: ${NELUX_CUDA_DLLS}")
      else()
        message(WARNING "[CUDA DEBUG] GLOB found NO cudart64_*.dll or nvrtc64_*.dll in ${CUDA_BIN_DIR}!")
      endif()
    else()
      message(WARNING "Could not determine CUDA BIN directory. CUDA DLLs will not be bundled!")
    endif()
  endif()

  if(NELUX_BUNDLE_FFMPEG_DLLS)
    # Bundle FFmpeg runtime DLLs from external/ffmpeg/bin when explicitly enabled.
    file(GLOB NELUX_FFMPEG_DLLS
      "${NELUX_FFMPEG_DIR}/bin/avcodec-*.dll"
      "${NELUX_FFMPEG_DIR}/bin/avformat-*.dll"
      "${NELUX_FFMPEG_DIR}/bin/avutil-*.dll"
      "${NELUX_FFMPEG_DIR}/bin/swscale-*.dll"
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
      "${VCPKG_LIB}/libyuv*.dylib"
      "${VCPKG_LIB}/libfmt*.dylib"
    )
  else()
    file(GLOB NELUX_SOS
      "${VCPKG_LIB}/libspdlog.so*"
      "${VCPKG_LIB}/libyuv.so*"
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
  if(NELUX_DLLS)
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
