cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")

set(CMAKE_PREFIX_PATH "/home/bozhan/libraries/libtorch")
set(CUFILE_LIBRARY "/usr/local/cuda/lib64/libcufile.so")

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(mscomp mscomp.cu)

target_link_libraries(mscomp ${TORCH_LIBRARIES})
target_link_libraries(mscomp ${CUFILE_LIBRARY})

target_link_libraries(mscomp OpenMP::OpenMP_CXX)
# target_compile_options(mscomp "${OpenMP_CXX}")
# SET(CMAKE_CUDA_FLAGS  "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX}")
# target_compile_options(mscomp "-fopenmp")

set_property(TARGET mscomp PROPERTY CXX_STANDARD 17)