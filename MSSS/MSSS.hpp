/*
 * Written by Quang Minh Bui(minhbq6@viettel.com.vn, buiquangminh1111@gmail.com)
 *
 *
    This software library implements CUDA version of the saliency detection algorithm
    described in

        "SALIENCY DETECTION USING MAXIMUM SYMMETRIC SURROUND"
        Radhakrishna Achanta and Sabine S ̈usstrunk.
        In Proceedings of 2010 IEEE 17th International Conference on Image Processing

    ----------------------------------------------------------------------
*/
#ifndef MSSS_HPP
#define MSSS_HPP
#include "define.h"
#include <math.h>
#include <string>
#include <stdint.h>
#include <thrust/device_vector.h>

class MSSS
{
public:
    MSSS(int width, int height);
    ~MSSS();
public:
    cv::cuda::GpuMat m_L;
    cv::cuda::GpuMat m_a;
    cv::cuda::GpuMat m_b;
    cv::cuda::GpuMat m_saliencyMap;
    cv::cuda::GpuMat m_integral_L;
    cv::cuda::GpuMat m_integral_a;
    cv::cuda::GpuMat m_integral_b;
    cv::cuda::GpuMat m_mask;
    int m_width;
    int m_height;

public:
    void gpuCreateBinaryMask(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, double threshold);
    void gpuAdd(cv::cuda::GpuMat src, double val);
    void gray2lab(cv::cuda::GpuMat src);
    void gpuGray2lab(unsigned char *src, unsigned char *L, unsigned char *a, unsigned char *b,
                     int width, int height);
    void getSaliencyMap();
    void gpuGetSaliencyMap(unsigned char * L, unsigned char * a, unsigned char *b,
                           int32_t *inteL, int32_t *intea, int32_t *inteb,
                           double *saliencyMap);
    std::vector<cv::Rect> detect(cv::cuda::GpuMat src);
    void gpuCopyNonContiguous(cv::cuda::GpuMat src, cv::cuda::GpuMat dst, int type);
};

#endif // MSSS_HPP
