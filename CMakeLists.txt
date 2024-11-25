cmake_minimum_required(VERSION 3.10)

project(ConvolutionLayer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 61)

add_executable(convolution_layer main.cu)

target_compile_options(convolution_layer PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
target_compile_options(convolution_layer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(convolution_layer PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(convolution_layer PRIVATE CUDA::cudart)
