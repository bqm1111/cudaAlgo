#ifndef CUDAUTILS_HPP
#define CUDAUTILS_HPP
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "opencv2/opencv.hpp"
#include "float.h"
#include <chrono>
#define threadsPerBlock 8
#define getMoment std::chrono::high_resolution_clock::now()
#define getTimeElapsed(end, start) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0

#define gpuErrChk(ans) {gpuAssert((ans), __FILE__, __LINE__);}
#define getExeTime(mess, ans) {auto start = getMoment;ans;auto end = getMoment; std::cout <<mess<< getTimeElapsed(end, start)<<std::endl;}

inline void gpuAssert(cudaError_t code, const char *file, int64_t line, bool abort = true)
{
    if(code!=cudaSuccess)
    {
        fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

#endif // CUDAUTILS_HPP
