cmake_minimum_required(VERSION 3.5)
set(CMAKE_CUDA_COMPILER /usr/local/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 CACHE STRING "CUDA architectures")
project(MSSS LANGUAGES CXX CUDA)
enable_language(CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/*.cpp
                  ${PROJECT_SOURCE_DIR}/*.cu)

add_executable(${PROJECT_NAME} ${SOURCES})
# cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -O3 -gencode arch=compute_61,code=sm_61)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 --compiler-options -fno-strict-aliasing --ptxas-options=-v --generate-line-info)
include_directories(/usr/local/cuda/include)
target_link_libraries(${PROJECT_NAME} -L/usr/local/cuda/lib64 -lcudart -lcuda)

# OpenCV
find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

