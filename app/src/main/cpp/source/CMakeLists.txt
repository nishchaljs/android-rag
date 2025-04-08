# Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required(VERSION 3.14)
project(android-faiss)
add_definitions(-w)

# Compiler flags for optimization
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -static-openmp -s -O3 -DSkip_f2c_Undefs -DNO_LONG_LONG -DNO_BLAS_WRAP")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -static-openmp -s -O3 -DSkip_f2c_Undefs -DNO_LONG_LONG -DNO_BLAS_WRAP")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a -mfpu=neon -mfloat-abi=softfp -DGGML_USE_NEON")

# Define project root
set(NATIVE_DIR ${CMAKE_SOURCE_DIR})
set(LLAMA_DIR ${NATIVE_DIR}/llama.cpp)  # CORRECTED path based on directory structure
set(FAISS_DIR ${NATIVE_DIR}/faiss)
set(EXTRA_DIR ${NATIVE_DIR}/extra)

# Adjust for different architectures
if ((${ANDROID_ABI} STREQUAL "arm64-v8a") OR (${ANDROID_ABI} STREQUAL "x86_64"))
    add_definitions(-DANDROID_BIT32)
else ()
    set(FAISS_DIR ${NATIVE_DIR}/source/faiss_32)
    add_definitions(-DANDROID_BIT64)
endif ()

# Define necessary macros before adding llama.cpp
add_definitions(
        -DGGML_USE_OPENMP
        -D_XOPEN_SOURCE=600
        -DNDEBUG
)

# Add options to disable unnecessary llama.cpp components
option(LLAMA_BUILD_SERVER "Build llama-server" OFF)
option(LLAMA_BUILD_EXAMPLES "Build llama examples" OFF)
option(LLAMA_NATIVE "Build with -march=native" OFF)
option(LLAMA_METAL "Build with Metal GPU acceleration" OFF)

# Collect faiss and extra module source files
file(GLOB EXTRA_FILES
        ${EXTRA_DIR}/clapack/SRC/*.c
        ${EXTRA_DIR}/clapack/SRC/*.cpp
)

file(GLOB FAISS_FILES
        ${FAISS_DIR}/*.cpp
)

# Include necessary directories
include_directories(
        ${FAISS_DIR}
        ${LLAMA_DIR}
        ${EXTRA_DIR}
        ${LLAMA_DIR}/ggml/include  # Add ggml include directory
)

# Create shared library
add_library(faiss SHARED
        native-lib.cpp
        ${EXTRA_FILES}
        ${FAISS_FILES}
        # Don't include llama files directly - we'll link to the llama target
)

add_library(common STATIC IMPORTED)
set_target_properties(common PROPERTIES
        IMPORTED_LOCATION ${LLAMA_DIR}/build/common/libcommon.a
)
add_library(ggml-base STATIC IMPORTED)
set_target_properties(ggml-base PROPERTIES
        IMPORTED_LOCATION ${LLAMA_DIR}/build/libggml-base.so
)
add_library(ggml-cpu STATIC IMPORTED)
set_target_properties(ggml-cpu PROPERTIES
        IMPORTED_LOCATION ${LLAMA_DIR}/build/libggml-cpu.so
)
add_library(ggml STATIC IMPORTED)
set_target_properties(ggml PROPERTIES
        IMPORTED_LOCATION ${LLAMA_DIR}/build/libggml.so
)
add_library(llama STATIC IMPORTED)
set_target_properties(llama PROPERTIES
        IMPORTED_LOCATION ${LLAMA_DIR}/build/libllama.so
)

# Find NDK log library
find_library(log-lib log)

# Link libraries including llama target
target_link_libraries(faiss
        ${log-lib}
        ggml-base
        ggml-cpu
        ggml
        llama  # Link to the llama target built by add_subdirectory
        common
)
