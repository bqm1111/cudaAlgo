#include <iostream>
#include <iterator>
#include <chrono>
#include <ostream>
#include <fstream>
#include <sys/stat.h>
#include <vector>
#include <dirent.h>
#include "multicue/multicue.hpp"
#include "CPU/cpumulticue.hpp"
#include "opencv/cv.hpp"
using namespace std;

int main(int argc, char* argv[])
{
    auto multiCue_bgs = multiCue::MultiCues();
    unsigned char * d_img;
    cv::VideoCapture cap(argv[1]);
    int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    gpuErrChk(cudaMalloc((void**)&d_img, sizeof(uchar) * width * height));
    cv::Mat frame;
    while(true)
    {
        cap.read(frame);
        if(frame.empty())
        {
            std::cout << "End of video" << std::endl;
        }
        cv::cvtColor(frame, frame, CV_BGR2GRAY);

        gpuErrChk(cudaMemcpy(d_img, frame.data, sizeof(uchar) * frame.rows * frame.cols, cudaMemcpyHostToDevice));
        getExeTime("Processing Time = ", multiCue_bgs.movingDetectObject(d_img, frame.data, frame.cols, frame.rows));

        std::cout << "Number of object found = " << multiCue_bgs.m_objRect.size() << std::endl;

        for(size_t i = 0; i < multiCue_bgs.m_objRect.size(); i++)
        {
            std::cout << "Drawing rect " << multiCue_bgs.m_objRect[i] << std::endl;
            cv::rectangle(frame,multiCue_bgs.m_objRect[i], cv::Scalar(0, 0, 0), 2, 4);
        }
        cv::imshow("frame", frame);
        cv::waitKey();
    }
    gpuErrChk(cudaFree(d_img));
}

