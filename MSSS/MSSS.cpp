#include "MSSS.hpp"
MSSS::MSSS(int width, int height)
{
    m_width = width;
    m_height = height;
    m_L = cv::cuda::createContinuous(height, width, CV_8UC1);
    m_a = cv::cuda::createContinuous(height, width, CV_8UC1);
    m_b = cv::cuda::createContinuous(height, width, CV_8UC1);
    m_integral_L = cv::cuda::createContinuous(height + 1, width + 1, CV_32SC1);
    m_integral_a = cv::cuda::createContinuous(height + 1, width + 1, CV_32SC1);
    m_integral_b = cv::cuda::createContinuous(height + 1, width + 1, CV_32SC1);
    m_mask = cv::cuda::createContinuous(height, width, CV_8UC1);
    m_saliencyMap = cv::cuda::createContinuous(height, width, CV_64F);
}

MSSS::~MSSS()
{}

void MSSS::gray2lab(cv::cuda::GpuMat src)
{
    gpuGray2lab(src.data, m_L.data, m_a.data, m_b.data, src.cols, src.rows);
}

void MSSS::getSaliencyMap()
{
    gpuGetSaliencyMap(m_L.data, m_a.data, m_b.data, (int32_t*)m_integral_L.data,(int32_t*)m_integral_a.data,
                      (int32_t*)m_integral_b.data, (double*)m_saliencyMap.data);
}

std::vector<cv::Rect> MSSS::detect(cv::cuda::GpuMat src)
{
    getExeTime("lab time = ", gray2lab(src));
    cv::cuda::GpuMat fuck;
    src.copyTo(m_L);
    getExeTime("integral time L= ", cv::cuda::integral(m_L, fuck));
    getExeTime("Copy integral time = ", gpuErrChk(cudaMemcpy2D((int32_t*)m_integral_L.data, sizeof(int32_t) * m_integral_L.cols, (int32_t*)fuck.data, fuck.step,
                           sizeof(int32_t) * m_integral_L.cols, m_integral_L.rows, cudaMemcpyDeviceToDevice)));

//    getExeTime("integral time a = ", cv::cuda::integral(m_a, m_integral_a));
//    getExeTime("integral time b = ", cv::cuda::integral(m_b, m_integral_b));
    getExeTime("get saliencyMap time = ", getSaliencyMap());
    double minVal, maxVal;
    cv::Mat test;
    getExeTime("down load sm time = ", m_saliencyMap.download(test));
//    std::cout << test << std::endl;
    cv::Point minLoc, maxLoc;
    getExeTime("minmaxLoc time = ", cv::cuda::minMaxLoc(m_saliencyMap, &minVal, &maxVal, &minLoc, &maxLoc));
    std::cout << "min - max = "<< minVal << " - " << maxVal << std::endl;
////    getExeTime("gpuAdd time = ", gpuAdd(m_saliencyMap, -minVal / (maxVal - minVal)));

    cv::Scalar mean, std;
    getExeTime("meanstd time = ", cv::meanStdDev(test, mean, std));
    double T = mean[0] + 2.55 * std[0];
    getExeTime("binary time = ", gpuCreateBinaryMask(m_saliencyMap, m_mask, T));

    cv::Mat mask;
    getExeTime("download mask time = ", m_mask.download(mask));
    cv::imshow("mask", mask);
    cv::Mat labels, stats, centroids;
//    double m =  cv::contourArea(bwimg);
    auto tic = std::chrono::high_resolution_clock::now();
//    cv::findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    getExeTime("Morphology time = ", cv::morphologyEx(mask, mask, cv::MORPH_OPEN,  kernel));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::imshow("after", mask);

    int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    auto toc = std::chrono::high_resolution_clock::now();
    std::cout << "Find contour time = : " << std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count()/1000.0 << std::endl;

    std::vector<cv::Rect> bboxes;
    for(int i = 1; i < n; i++)
    {
        cv::Rect rr = cv::Rect(stats.at<int>(i, 0), stats.at<int>(i, 1),
                               stats.at<int>(i, 2), stats.at<int>(i, 3));
        
        int area = rr.width * rr.height;
        double ratio = (double) rr.width / (double) rr.height ;
        if(area > 140 && ratio > 0.08 && ratio < 13)
        {
            bboxes.push_back(rr);
//            cv::Mat candidateship = im(rr);
//            cv::imshow("candidate", candidateship);
//            cv::waitKey(30);
        }
    }

    return bboxes;
}

