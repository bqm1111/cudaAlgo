#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "warper.hpp"
#include <chrono>

using namespace std;

int main()
{
    int focal_length = 920;
    cv::Mat img = cv::imread("/home/martin/Pictures/thermal_radar/crop4.png", 0);
    cv::Mat K = (cv::Mat_<float>(3, 3) << focal_length, 0, img.cols / 2, 0, focal_length, img.rows / 2, 0, 0, 1);
    cv::Mat R = cv::Mat::eye(3, 3, CV_32F);
    cv::cuda::GpuMat d_img;
    d_img.upload(img);
    cv::cuda::GpuMat d_res, own_res;
    cv::Mat res, myres;

    // Get result from opencv function
    cv::Ptr<cv::detail::RotationWarper> m_warper;
    cv::Point corner;
    //    cv::detail::SphericalWarperGpu w(focal_length);
    //    auto start = std::chrono::high_resolution_clock::now();
    //    corner = w.warp(d_img, K, R, cv::INTER_LINEAR, cv::BORDER_REFLECT, d_res);
    //    auto end = std::chrono::high_resolution_clock::now();
    //    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0 << std::endl;

    //    d_res.download(res);
    //    cv::imshow("res", res);

    // Get result from my implementation

    Warper::SphericalWarperGpu warper(focal_length);
    auto start = std::chrono::high_resolution_clock::now();
    corner = warper.warp(d_img, K, R, cv::INTER_LINEAR, cv::BORDER_REFLECT, own_res);

    for (int i = 0; i < 1000; i++)
        corner = warper.warp(d_img, K, R, cv::INTER_LINEAR, cv::BORDER_REFLECT, own_res);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0 / 1000.0 << std::endl;
    own_res.download(myres);
    //    cv::imshow("own res", myres);
    //    cv::waitKey(0);
    return 0;
}
