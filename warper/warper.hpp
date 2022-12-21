/*
 * This software library implements the warping transformation from an image plane to
 * a spherical coordinate using CUDA
*/

#ifndef WARPER_HPP
#define WARPER_HPP
#include "opencv4/opencv2/opencv.hpp"
#include <iostream>
#include "cudaWarper.hpp"
#include "opencv4/opencv2/cudawarping.hpp"

namespace Warper {


struct ProjectBase
{
    void setCameraParams(cv::InputArray _K = cv::Mat::eye(3,3, CV_32F),
                         cv::InputArray _R = cv::Mat::eye(3,3, CV_32F),
                         cv::InputArray _T = cv::Mat::eye(3,1, CV_32F));
    float scale;
    float k[9];
    float rinv[9];
    float r_kinv[9];
    float k_rinv[9];
    float t[3];
};

struct SphericalProjector : ProjectBase
{
    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);
};

class SphericalWarperGpu
{
public:
    SphericalWarperGpu(float scale) {projector_.scale = scale;}
    cv::Rect buildMaps(cv::Size src_size, cv::Mat K, cv::Mat R, cv::Mat xmap, cv::Mat ymap);
    cv::Rect buildMaps(cv::Size src_size, cv::Mat K, cv::Mat R, cv::cuda::GpuMat &xmap, cv::cuda::GpuMat &ymap);
    cv::Rect buildMaps(cv::Size src_size, cv::Mat K, cv::Mat R, float * xmap, float *ymap);

    cv::Point warp(cv::cuda::GpuMat &src, cv::Mat K, cv::Mat R,
                   int interp_mode, int border_mode,
                   cv::cuda::GpuMat &dst);
    cv::Point warp(unsigned char *src, cv::Mat K, cv::Mat R,
                   int interp_mode, int border_mode, unsigned char *dst);
protected:
    void buildWarpSphericalMaps(cv::Size src_size, cv::Rect dst_roi, cv::Mat K, cv::Mat R,
                                       float scale, cv::cuda::GpuMat &map_x, cv::cuda::GpuMat &map_y);
    void buildWarpSphericalMaps(cv::Size src_size, cv::Rect dst_roi, cv::Mat K, cv::Mat R,
                                float scale, float *map_x, float * map_y);

    void detectResultRoi(cv::Size src_size, cv::Point &dst_tl, cv::Point &dst_br);
    void detectResultRoiByBorder(cv::Size src_size, cv::Point &dst_tl, cv::Point &dst_br);

    SphericalProjector projector_;

private:
    cv::cuda::GpuMat d_xmap_, d_ymap_, d_src_, d_dst_;
};
}

#endif // WARPER_HPP
