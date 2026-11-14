cmake_minimum_required(VERSION 3.18)
project(nexus VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build type ──────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags (platform-aware) ─────────────────────────────────────────
if(MSVC)
    # MSVC / clang-cl on Windows
    set(NEXUS_CXX_FLAGS
        /O2                        # Optimise (equiv -O2; MSVC has no -O3)
        /arch:AVX2                 # AVX2 — matches -march=native on modern x86
        /fp:fast                   # Fast math
        /W3                        # Warnings
        /wd4244 /wd4267 /wd4996    # Suppress common narrowing + deprecated warnings
        /MP                        # Parallel compilation
        /D_CRT_SECURE_NO_WARNINGS
        /DNOMINMAX
    )
else()
    # GCC / Clang (Linux, macOS, MinGW on Windows)
    # ── CPU tier ──────────────────────────────────────────────────────────────
    # The engine is built ONCE PER TIER (same source, different -march); the tiny
    # `nexus_server` dispatcher (tools/nexus_dispatch.cpp, built with NO -march) picks the
    # best one at startup.  Names are compiler-version independent on purpose.
    #   v2     -march=nehalem          SSE4.2 + POPCNT          every x86 CPU since 2009
    #   v3     -march=haswell          AVX2+FMA+BMI2+F16C       Haswell 2013+, Zen 1+  (default)
    #   v4     -march=skylake-avx512   AVX-512 F/BW/DQ/VL       Skylake-X, Zen 4+      (optional)
    #   native -march=native           THIS machine only: for local testing, never ship it
    set(NEXUS_TIER "v3" CACHE STRING "CPU tier to build: v2 | v3 | v4 | native")
    set_property(CACHE NEXUS_TIER PROPERTY STRINGS v2 v3 v4 native)
    if(NEXUS_TIER STREQUAL "v2")
        set(MARCH_FLAG -march=nehalem -mtune=generic)
    elseif(NEXUS_TIER STREQUAL "v3")
        set(MARCH_FLAG -march=haswell -mtune=generic)
    elseif(NEXUS_TIER STREQUAL "v4")
        set(MARCH_FLAG -march=skylake-avx512 -mtune=generic)
    elseif(NEXUS_TIER STREQUAL "native")
        set(MARCH_FLAG -march=native)
    else()
        message(FATAL_ERROR "NEXUS_TIER must be v2, v3, v4 or native (got '${NEXUS_TIER}')")
    endif()
    message(STATUS "CPU tier: ${NEXUS_TIER}  (${MARCH_FLAG})")
    set(NEXUS_CXX_FLAGS
        -O3
        ${MARCH_FLAG}
        -ffast-math
        -funroll-loops
        -fomit-frame-pointer
        -Wall -Wextra
        -Wno-unused-parameter
    )
endif()

# ── CUDA support (optional) ───────────────────────────────────────────────────
option(NEXUS_CUDA "Build with CUDA GPU support" OFF)
option(NEXUS_ROCM "Build with AMD ROCm/HIP support" OFF)
option(NEXUS_CUDA_ARCH "CUDA compute architectures" "")

if(NEXUS_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(NEXUS_CXX_FLAGS ${NEXUS_CXX_FLAGS} -DNEXUS_CUDA)
    message(STATUS "CUDA enabled")
    if(NEXUS_CUDA_ARCH)
        set(CMAKE_CUDA_ARCHITECTURES ${NEXUS_CUDA_ARCH})
        message(STATUS "CUDA architectures: ${NEXUS_CUDA_ARCH}")
    endif()
endif()

if(NEXUS_ROCM)
    find_package(hip REQUIRED)
    set(NEXUS_CXX_FLAGS ${NEXUS_CXX_FLAGS} -DNEXUS_ROCM)
    message(STATUS "ROCm/HIP enabled")
endif()

# ── Vulkan compute support (integrated GPU / AMD RADV) ────────────────────────
option(NEXUS_VULKAN "Build with Vulkan compute GPU support" OFF)
if(NEXUS_VULKAN)
    # Vulkan is loaded at RUNTIME (src/gpu/vk_loader.h: dlopen / LoadLibrary), so the
    # binary has no link-time dependency on libvulkan / vulkan-1.dll and still starts
    # on machines without a Vulkan stack (CPU fallback).  We only need the HEADERS here.
    find_package(Vulkan QUIET)
    if(TARGET Vulkan::Headers)
        set(NEXUS_VK_INCLUDE_TARGET Vulkan::Headers)
    elseif(Vulkan_INCLUDE_DIRS)
        set(NEXUS_VK_INCLUDE_DIRS ${Vulkan_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "NEXUS_VULKAN=ON needs the Vulkan headers "
                "(libvulkan-dev / Vulkan SDK / Vulkan-Headers); the loader library is NOT required.")
    endif()
    set(NEXUS_CXX_FLAGS ${NEXUS_CXX_FLAGS} -DNEXUS_VULKAN -DVK_NO_PROTOTYPES)
    message(STATUS "Vulkan enabled (runtime-loaded; headers only)")

    # ── GPU shaders -> SPIR-V -> shader_spirv.h ──────────────────────────────
    # Generated into the BUILD tree (never src/), by tools/gen_shaders.py, so a stale
    # or copied header in src/ can no longer shadow it, and the rebuild triggers on
    # ANY .comp or script change.  A missing shader compiler is a hard error: building
    # without shaders would silently produce an engine whose GPU path does nothing.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    find_program(GLSLC glslc)
    find_program(GLSLANG glslangValidator)
    if(NOT GLSLC AND NOT GLSLANG)
        message(FATAL_ERROR "NEXUS_VULKAN=ON needs a shader compiler: glslc (shaderc) or "
                "glslangValidator. Ubuntu: apt install glslc   MSYS2: pacman -S mingw-w64-x86_64-shaderc")
    endif()
    file(GLOB NEXUS_SHADER_SRCS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/shaders/*.comp)
    set(NEXUS_SHADER_HEADER ${CMAKE_BINARY_DIR}/gen/nexus_shader_spirv.h)
    add_custom_command(
        OUTPUT  ${NEXUS_SHADER_HEADER}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/gen_shaders.py
                --shaders-dir ${CMAKE_SOURCE_DIR}/src/gpu/shaders
                --out ${NEXUS_SHADER_HEADER}
                "--glslc=${GLSLC}" "--glslang=${GLSLANG}"
        DEPENDS ${NEXUS_SHADER_SRCS} ${CMAKE_SOURCE_DIR}/tools/gen_shaders.py
        COMMENT "Compiling GPU shaders..."
        VERBATIM)
    add_custom_target(nexus_shaders DEPENDS ${NEXUS_SHADER_HEADER})
    set(NEXUS_CXX_FLAGS ${NEXUS_CXX_FLAGS} -DNEXUS_SHADERS_EMBEDDED)
    message(STATUS "Shader compiler: ${GLSLC}${GLSLANG}")
    # The backend includes "nexus_shader_spirv.h" (a new name), so an old
    # src/gpu/shader_spirv.h can no longer be picked up by mistake.
    if(EXISTS ${CMAKE_SOURCE_DIR}/src/gpu/shader_spirv.h)
        message(STATUS "Note: src/gpu/shader_spirv.h is an unused leftover (safe to delete); "
                       "the build now generates nexus_shader_spirv.h in ${CMAKE_BINARY_DIR}/gen.")
    endif()
endif()

# ── OpenBLAS for CPU matmul ───────────────────────────────────────────────────
#find_package(BLAS)
#if(BLAS_FOUND)
#    set(NEXUS_CXX_FLAGS ${NEXUS_CXX_FLAGS} -DNEXUS_BLAS)
#    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
#endif()

# ── Source files ──────────────────────────────────────────────────────────────
#
# Architecture note:
#   quant, matmul, attention, kvcache, transformer are header-only (.h files
#   contain both declaration and implementation). This is intentional — it lets
#   the compiler inline aggressively across the whole forward pass.
#
#   engine.cpp is the single compilation unit that #includes all of them,
#   plus it provides the public Engine API and chains everything together.
#
#   memory.cpp and tokenizer.cpp are traditional split files (too large
#   to be header-only without hurting compile times).
#
set(NEXUS_SOURCES
    src/core/memory.cpp          # memory manager (huge pages, mlock, KV paging)
    src/model/tokenizer.cpp      # BPE tokenizer (SentencePiece + GPT-2 styles)
    src/api/engine.cpp           # public API — includes all header-only modules
)

if(NEXUS_CUDA)
    list(APPEND NEXUS_SOURCES
        src/cuda/matmul.cu
        src/cuda/attention.cu
        src/cuda/transfer.cu
    )
endif()

# ── Main library ──────────────────────────────────────────────────────────────
add_library(nexus SHARED ${NEXUS_SOURCES})
if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
# Portable build → libnexus.dll / libnexus.so   (runs on all CPUs)
# Native build   → libnexusnative.dll / libnexusnative.so  (fast, modern CPUs only)
# Both live in the same folder alongside their matching server binary.
# Library name carries the tier:  libnexus_v3.so / nexus_v3.dll ... The matching
# nexus_server_<tier> is linked against it and finds it next to itself.
if(NOT NEXUS_TIER)
    set(NEXUS_TIER "v3")   # MSVC branch: single AVX2 build
endif()
set_target_properties(nexus PROPERTIES OUTPUT_NAME "nexus_${NEXUS_TIER}")
target_compile_options(nexus PRIVATE ${NEXUS_CXX_FLAGS})
if(NEXUS_VULKAN)
    target_include_directories(nexus PUBLIC ${CMAKE_BINARY_DIR}/gen)   # generated shader_spirv.h
endif()
target_include_directories(nexus PUBLIC src/)

#if(BLAS_FOUND)
#    target_link_libraries(nexus PUBLIC ${BLAS_LIBRARIES})
#endif()

if(NEXUS_VULKAN)
    if(NEXUS_VK_INCLUDE_TARGET)
        target_link_libraries(nexus PUBLIC ${NEXUS_VK_INCLUDE_TARGET})
    else()
        target_include_directories(nexus PUBLIC ${NEXUS_VK_INCLUDE_DIRS})
    endif()
    target_link_libraries(nexus PUBLIC ${CMAKE_DL_LIBS})   # dlopen
    if(TARGET nexus_shaders)
        add_dependencies(nexus nexus_shaders)
    endif()
endif()

if(NEXUS_CUDA)
    target_link_libraries(nexus PUBLIC CUDA::cudart CUDA::cublas)
endif()

# ── Test / diagnostic binary ──────────────────────────────────────────────────
add_executable(nexus_inspect tools/inspect.cpp)
target_compile_options(nexus_inspect PRIVATE ${NEXUS_CXX_FLAGS})
target_link_libraries(nexus_inspect nexus)
target_include_directories(nexus_inspect PRIVATE src/)

add_executable(nexus_run tools/run.cpp)
target_compile_options(nexus_run PRIVATE ${NEXUS_CXX_FLAGS})
target_link_libraries(nexus_run nexus)
target_include_directories(nexus_run PRIVATE src/)

add_executable(nexus_server tools/server.cpp)
set_target_properties(nexus_server PROPERTIES OUTPUT_NAME "nexus_server_${NEXUS_TIER}")
target_compile_options(nexus_server PRIVATE ${NEXUS_CXX_FLAGS})
if(WIN32)
    target_link_libraries(nexus_server nexus ws2_32 psapi)
    # Static link runtimes — no libstdc++-6.dll needed on target machine
    # Only apply to nexus_server — applying to the shared nexus DLL causes
    # duplicate _Unwind_Resume definition at link time
    target_link_options(nexus_server PRIVATE -static-libgcc -static-libstdc++)
else()
    target_link_libraries(nexus_server nexus pthread)
endif()
target_include_directories(nexus_server PRIVATE src/)

# Find libnexus*.so / .dylib NEXT TO the executable, wherever the folder is copied
# (USB stick, another machine).  Without this the binary only finds the library at the
# absolute build path or via LD_LIBRARY_PATH.
if(UNIX AND NOT APPLE)
    set_target_properties(nexus_server PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    set_target_properties(nexus_server PROPERTIES BUILD_WITH_INSTALL_RPATH ON INSTALL_RPATH "@executable_path")
endif()

# ── Dispatcher: the `nexus_server` users start ────────────────────────────────
# Compiled WITHOUT any -march so it runs on every CPU, statically linked so it needs
# nothing installed.  It picks and starts nexus_server_<tier> (see its header comment).
add_executable(nexus_dispatch tools/nexus_dispatch.cpp)
set_target_properties(nexus_dispatch PROPERTIES OUTPUT_NAME "nexus_server")
target_compile_options(nexus_dispatch PRIVATE -O2 -Wall -Wextra)
if(WIN32)
    target_link_options(nexus_dispatch PRIVATE -static)
else()
    target_link_options(nexus_dispatch PRIVATE -static-libgcc -static-libstdc++)
endif()

# nexus_server also needs shaders before it can build (now that nexus_server exists)
if(NEXUS_VULKAN AND TARGET nexus_shaders)
    add_dependencies(nexus_server nexus_shaders)
endif()

# ── Python bindings (pybind11) ────────────────────────────────────────────────
option(NEXUS_PYTHON "Build Python bindings" OFF)
if(NEXUS_PYTHON)
    find_package(Python 3.8 COMPONENTS Interpreter Development)
    if(Python_FOUND)
        # Try cmake config first, then fall back to python -m pybind11 --cmakedir
        find_package(pybind11 CONFIG QUIET)
        if(NOT pybind11_FOUND)
            execute_process(
                COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(PYBIND11_CMAKE_DIR)
                find_package(pybind11 CONFIG QUIET PATHS ${PYBIND11_CMAKE_DIR} NO_DEFAULT_PATH)
            endif()
        endif()

        if(pybind11_FOUND)
            pybind11_add_module(nexus_py src/api/python.cpp)
            target_link_libraries(nexus_py PRIVATE nexus)
            target_include_directories(nexus_py PRIVATE src/)
            message(STATUS "Python bindings enabled (pybind11 ${pybind11_VERSION})")
        else()
            message(STATUS "pybind11 not found — Python bindings skipped")
            message(STATUS "  Install: pip install pybind11 --break-system-packages")
        endif()
    endif()
endif()

# ── Install (run separately: sudo make install) ───────────────────────────────
# Not part of default 'make' — run explicitly with: cd build && sudo make install
install(TARGETS nexus            DESTINATION lib)
install(TARGETS nexus_inspect    DESTINATION bin)
install(TARGETS nexus_run        DESTINATION bin)
install(TARGETS nexus_server nexus_dispatch DESTINATION bin)

# ── Release: strip debug symbols ────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        # MSVC: no PDB in release — symbols already absent with /O2 without /Zi
        target_compile_options(nexus        PRIVATE /Gy)   # function-level linking
        target_compile_options(nexus_server PRIVATE /Gy)
    elseif(APPLE)
        # macOS: strip via linker flag
        target_link_options(nexus_server PRIVATE -Wl,-x)
    else()
        # Linux: strip all symbols
        if(NOT WIN32)
            target_link_options(nexus_server PRIVATE -s)
        endif()
    endif()
endif()

# ── Build summary ─────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "── Nexus Build Configuration ───────────────")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA:  ${NEXUS_CUDA}")
message(STATUS "  ROCm:  ${NEXUS_ROCM}")
message(STATUS "  BLAS:  ${BLAS_FOUND}")
message(STATUS "  Python bindings: ${NEXUS_PYTHON}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}  Compiler: ${CMAKE_CXX_COMPILER_ID}  Tier: ${NEXUS_TIER}")
message(STATUS "────────────────────────────────────────────")
message(STATUS "")
