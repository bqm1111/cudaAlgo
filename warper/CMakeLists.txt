cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.5/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 75 CACHE STRING "CUDA architectures")

project(experiment LANGUAGES CXX CUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fno-omit-frame-pointer -fsanitize=leak -g -pthread")

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
find_package(spdlog REQUIRED)

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/*.cpp
                    ${PROJECT_SOURCE_DIR}/*.cu
                    )
                    
add_executable(${PROJECT_NAME} ${SOURCES})
pkg_check_modules(Uuid REQUIRED IMPORTED_TARGET uuid)
include_directories(${Uuid_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${Uuid_LIBRARIES})

# cuda
find_package(CUDA REQUIRED)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --compiler-options -fno-strict-aliasing --ptxas-options=-v --generate-line-info")
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} -lstdc++fs -lcurl)

