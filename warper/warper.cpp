#include "warper.hpp"

namespace cudaMapper {
void buildWarpSphericalMaps(int tl_u, int tl_v,
                            float * map_x, float * map_y,
                            int map_rows, int map_cols,
                            const float k_rinv[9], const float r_kinv[9], float scale);

}

namespace Warper {
void ProjectBase::setCameraParams(cv::InputArray _K, cv::InputArray _R, cv::InputArray _T)
{
    cv::Mat K = _K.getMat(), R = _R.getMat(), T = _T.getMat();

    CV_Assert(K.size() == cv::Size(3, 3) && K.type() == CV_32F);
    CV_Assert(R.size() == cv::Size(3, 3) && R.type() == CV_32F);
    CV_Assert((T.size() == cv::Size(1, 3) || T.size() == cv::Size(3, 1)) && T.type() == CV_32F);

    cv::Mat_<float> K_(K);
    k[0] = K_(0,0); k[1] = K_(0,1); k[2] = K_(0,2);
    k[3] = K_(1,0); k[4] = K_(1,1); k[5] = K_(1,2);
    k[6] = K_(2,0); k[7] = K_(2,1); k[8] = K_(2,2);

    cv::Mat_<float> Rinv = R.t();
    rinv[0] = Rinv(0,0); rinv[1] = Rinv(0,1); rinv[2] = Rinv(0,2);
    rinv[3] = Rinv(1,0); rinv[4] = Rinv(1,1); rinv[5] = Rinv(1,2);
    rinv[6] = Rinv(2,0); rinv[7] = Rinv(2,1); rinv[8] = Rinv(2,2);

    cv::Mat_<float> R_Kinv = R * K.inv();
    r_kinv[0] = R_Kinv(0,0); r_kinv[1] = R_Kinv(0,1); r_kinv[2] = R_Kinv(0,2);
    r_kinv[3] = R_Kinv(1,0); r_kinv[4] = R_Kinv(1,1); r_kinv[5] = R_Kinv(1,2);
    r_kinv[6] = R_Kinv(2,0); r_kinv[7] = R_Kinv(2,1); r_kinv[8] = R_Kinv(2,2);

    cv::Mat_<float> K_Rinv = K * Rinv;
    k_rinv[0] = K_Rinv(0,0); k_rinv[1] = K_Rinv(0,1); k_rinv[2] = K_Rinv(0,2);
    k_rinv[3] = K_Rinv(1,0); k_rinv[4] = K_Rinv(1,1); k_rinv[5] = K_Rinv(1,2);
    k_rinv[6] = K_Rinv(2,0); k_rinv[7] = K_Rinv(2,1); k_rinv[8] = K_Rinv(2,2);

    cv::Mat_<float> T_(T.reshape(0, 3));
    t[0] = T_(0,0); t[1] = T_(1,0); t[2] = T_(2,0);
}

inline void SphericalProjector::mapForward(float x, float y, float &u, float &v)
{
    float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
    float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
    float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

    u = scale * atan2(x_, z_);
    float w = y_ / sqrtf(x_ * x_ + y_ * y_ + z_ * z_);
    v = scale * (static_cast<float>(M_PI) - acosf(w == w ? w : 0));
}

inline void SphericalProjector::mapBackward(float u, float v, float &x, float &y)
{
    u /= scale;
    v /= scale;
    float sinv = sinf(static_cast<float>(M_PI) - v);
    float x_ = sinv * sinf(u);
    float y_ = cosf(static_cast<float>(M_PI) - v);
    float z_ = sinv * cosf(u);

    float z;
    x = k_rinv[0] * x_ + k_rinv[1] * y_ + k_rinv[2] * z_;
    y = k_rinv[3] * x_ + k_rinv[4] * y_ + k_rinv[5] * z_;
    z = k_rinv[6] * x_ + k_rinv[7] * y_ + k_rinv[8] * z_;
    if(z > 0)
    {
        x /= z;
        y /= z;
    }
    else
        x = y = -1;
}


void SphericalWarperGpu::detectResultRoi(cv::Size src_size, cv::Point &dst_tl, cv::Point &dst_br)
{
    detectResultRoiByBorder(src_size, dst_tl, dst_br);
    float tl_uf = static_cast<float>(dst_tl.x);
    float tl_vf = static_cast<float>(dst_tl.y);
    float br_uf = static_cast<float>(dst_br.x);
    float br_vf = static_cast<float>(dst_br.y);

    float x = projector_.rinv[1];
    float y = projector_.rinv[4];
    float z = projector_.rinv[7];
    if (y > 0.f)
    {
        float x_ = (projector_.k[0] * x + projector_.k[1] * y) / z + projector_.k[2];
        float y_ = projector_.k[4] * y / z + projector_.k[5];
        if (x_ > 0.f && x_ < src_size.width && y_ > 0.f && y_ < src_size.height)
        {
            tl_uf = std::min(tl_uf, 0.f); tl_vf = std::min(tl_vf, static_cast<float>(CV_PI * projector_.scale));
            br_uf = std::max(br_uf, 0.f); br_vf = std::max(br_vf, static_cast<float>(CV_PI * projector_.scale));
        }
    }

    x = projector_.rinv[1];
    y = -projector_.rinv[4];
    z = projector_.rinv[7];
    if (y > 0.f)
    {
        float x_ = (projector_.k[0] * x + projector_.k[1] * y) / z + projector_.k[2];
        float y_ = projector_.k[4] * y / z + projector_.k[5];
        if (x_ > 0.f && x_ < src_size.width && y_ > 0.f && y_ < src_size.height)
        {
            tl_uf = std::min(tl_uf, 0.f); tl_vf = std::min(tl_vf, static_cast<float>(0));
            br_uf = std::max(br_uf, 0.f); br_vf = std::max(br_vf, static_cast<float>(0));
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


void SphericalWarperGpu::detectResultRoiByBorder(cv::Size src_size, cv::Point &dst_tl, cv::Point &dst_br)
{
    float tl_uf = (std::numeric_limits<float>::max)();
    float tl_vf = (std::numeric_limits<float>::max)();
    float br_uf = -(std::numeric_limits<float>::max)();
    float br_vf = -(std::numeric_limits<float>::max)();

    float u, v;
    for (float x = 0; x < src_size.width; ++x)
    {
        projector_.mapForward(static_cast<float>(x), 0, u, v);
        tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
        br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);

        projector_.mapForward(static_cast<float>(x), static_cast<float>(src_size.height - 1), u, v);
        tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
        br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);
    }
    for (int y = 0; y < src_size.height; ++y)
    {
        projector_.mapForward(0, static_cast<float>(y), u, v);
        tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
        br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);

        projector_.mapForward(static_cast<float>(src_size.width - 1), static_cast<float>(y), u, v);
        tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
        br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


cv::Rect SphericalWarperGpu::buildMaps(cv::Size src_size, cv::Mat K, cv::Mat R, cv::cuda::GpuMat &xmap, cv::cuda::GpuMat &ymap)
{
    projector_.setCameraParams(K, R);
    cv::Point dst_tl, dst_br;
    detectResultRoi(src_size, dst_tl, dst_br);
    buildWarpSphericalMaps(src_size, cv::Rect(dst_tl, cv::Point(dst_br.x + 1, dst_br.y + 1)),
                           K, R, projector_.scale, xmap, ymap);
    return cv::Rect(dst_tl, dst_br);
}

cv::Rect SphericalWarperGpu::buildMaps(cv::Size src_size, cv::Mat K, cv::Mat R, float *xmap, float *ymap)
{
    projector_.setCameraParams(K, R);
    cv::Point dst_tl, dst_br;
    detectResultRoi(src_size, dst_tl, dst_br);
    buildWarpSphericalMaps(src_size, cv::Rect(dst_tl, cv::Point(dst_br.x + 1, dst_br.y + 1)),
                           K, R, projector_.scale, xmap, ymap);
    return cv::Rect(dst_tl, dst_br);
}

void SphericalWarperGpu::buildWarpSphericalMaps(cv::Size src_size, cv::Rect dst_roi, cv::Mat K, cv::Mat R, float scale, float *map_x, float *map_y)
{
    assert(K.size() == cv::Size(3,3) && K.type() == CV_32F);
    assert(R.size() == cv::Size(3,3) && R.type() == CV_32F);
    //    assert(T.size() == cv::Size(3,1) || T.size()== cv::Size(1,3));

    cv::Mat K_Rinv = K * R.t();
    cv::Mat R_Kinv = R * K.inv();

    cudaMalloc((void**)& map_x, dst_roi.height * dst_roi.width * sizeof(float));
    cudaMalloc((void **)&map_y, dst_roi.height * dst_roi.width * sizeof(float));

    cudaMapper::buildWarpSphericalMaps(dst_roi.tl().x, dst_roi.tl().y, map_x, map_y, dst_roi.height, dst_roi.width, K_Rinv.ptr<float>(),
                                       R_Kinv.ptr<float>(), scale);
}

void SphericalWarperGpu::buildWarpSphericalMaps(cv::Size src_size, cv::Rect dst_roi, cv::Mat K, cv::Mat R,
                                   float scale, cv::cuda::GpuMat &map_x, cv::cuda::GpuMat &map_y)

{
    assert(K.size() == cv::Size(3,3) && K.type() == CV_32F);
    assert(R.size() == cv::Size(3,3) && R.type() == CV_32F);
    //    assert(T.size() == cv::Size(3,1) || T.size()== cv::Size(1,3));

    cv::Mat K_Rinv = K * R.t();
    cv::Mat R_Kinv = R * K.inv();

    map_x = cv::cuda::createContinuous(dst_roi.size(),CV_32F);
    map_y = cv::cuda::createContinuous(dst_roi.size(), CV_32F);

    cudaMapper::buildWarpSphericalMaps(dst_roi.tl().x, dst_roi.tl().y, map_x.ptr<float>(), map_y.ptr<float>(), map_x.rows, map_x.cols, K_Rinv.ptr<float>(),
                                       R_Kinv.ptr<float>(), scale);
}


cv::Point SphericalWarperGpu::warp(cv::cuda::GpuMat &src, cv::Mat K, cv::Mat R, int interp_mode, int border_mode, cv::cuda::GpuMat &dst)
{
    cv::Rect dst_roi = buildMaps(src.size(), K, R, d_xmap_, d_ymap_);
    dst = cv::cuda::createContinuous(dst_roi.height + 1, dst_roi.width + 1, src.type());
//    dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());
    cv::cuda::remap(src, dst, d_xmap_, d_ymap_, interp_mode, border_mode);
    return dst_roi.tl();
}

cv::Point SphericalWarperGpu::warp(unsigned char *src, cv::Mat K, cv::Mat R, int interp_mode, int border_mode, unsigned char *dst)
{
}
}
