#ifndef DEFINE_H
#define DEFINE_H

#include <iostream>
#include <thread>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <climits>
#include <cfloat>
#include <cmath>
#include <vector>
#include <limits>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cufft.h"
#include "cuda_runtime_api.h"
#include "cooperative_groups.h"
#include <stdlib.h>
#include <assert.h>
#include "opencv2/opencv.hpp"
#include "cusolverDn.h"
#include "cudnn.h"
#include "sys/time.h"

#define TEST

#define SHOWTIME
#define threadsPerBlock 16
#define IMUL(a, b) __mul24(a, b)
#ifdef SHOWTIME
#define getKernelTime(mess, end, start) {double time = (1000000.0 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec) / 1000.0;printf(mess, time);}
#define getMoment std::chrono::high_resolution_clock::now()
#define getTimeElapsed(mess, end, start) {std::cout << mess << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0 << std::endl;}
#define getExeTime(mess, ans) {auto start = getMoment;ans;auto end = getMoment; getTimeElapsed(mess, end, start);}
#define testCUDATime(mess, iteration, ans)  {auto start = getMoment; for(int i = 0; i < iteration; i++){ans;} auto end = getMoment; \
    std::cout << mess << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0 / iteration << std::endl;}
#else
#define getMoment 1
#define getTimeElapsed(mess, end, start)    {}
#define getExeTime(mess, ans)   {ans;}
#define getKernelTime(mess, end, start) {}

#endif
#define gpuErrChk(ans) {gpuAssert((ans), __FILE__, __LINE__);}

#ifndef uchar
typedef unsigned char uchar;
#endif
inline void gpuAssert(cudaError_t code, const char *file, int64_t line, bool abort = true)
{
    if(code!=cudaSuccess)
    {
        fprintf(stderr, "GPUAssert: %s %s %ld\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

/*********************/
/* CUFFT ERROR CHECK */
/*********************/
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch(error)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
        return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:
        return "CUFFT_LICENSE_ERROR";
    case CUFFT_NOT_SUPPORTED:
        return "CUFFT_NOT_SUPPORTED";
    }
    return "<unknown>";
}

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, int64_t line)
{
    if( CUFFT_SUCCESS != err) {
        fprintf(stderr, "CUFFT error: %s %s in line %ld", _cudaGetErrorEnum(err), file, line);
        cudaDeviceReset(); assert(0);
    }
}

#define LOG_MSG( ... )      {printf( __VA_ARGS__ ); exit(0);}
#define DEVICE_LOG_MSG( ... ) printf( __VA_ARGS__ )
//#define printTexture
//#define printColor

#endif // DEFINE_H
