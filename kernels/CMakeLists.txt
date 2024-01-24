cmake_minimum_required(VERSION 3.5)
project (maxk_kernel_test C CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(${PROJECT_NAME} main.cu spmm_maxk.cu spmm_maxk_backward.cu spmm_cusparse.cu)

# add_executable(${PROJECT_NAME} maxk_kernel.cu)

target_link_libraries(${PROJECT_NAME}  cusparse stdc++fs)