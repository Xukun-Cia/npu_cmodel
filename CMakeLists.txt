cmake_minimum_required(VERSION 3.10)

project(npu_cmodel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# Source files
set(SOURCES
    npu_top.cpp
    npu_sram.cpp
    npu_dma.cpp
    npu_cpu.cpp
    npu_rvv.cpp
    npu_tpu.cpp
    usecase/t000_silu.cpp
    operator/operator_silu.cpp
    operator/operator_matmul.cpp
    usecase/t001_matmul.cpp
    operator/operator_gelu.cpp
    usecase/t002_gelu.cpp
    operator/operator_conv3d.cpp
    usecase/t003_conv3d.cpp
    operator/operator_conv3d_patched.cpp
    usecase/t004_conv3d_patched.cpp
)

# Header path
include_directories(.)
include_directories(usecase)
include_directories(operator)

# Executable
add_executable(npu_top ${SOURCES})

# Compiler flags (optional)
if(MSVC)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
endif()

