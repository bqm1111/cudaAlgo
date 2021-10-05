#include "test.hpp"

void test()
{
    int src[40];
    for(int i = 0; i < 40; i++)
    {
        src[i] = i + 1;
//        std::cout << (int)src[i] << std::endl;
    }
    std::cout << "size of int = " << sizeof(int) << std::endl;
    std::cout << (int)(*(src + 10)) << std::endl;
    std::cout << (int)*((unsigned char*)src + 40) << std::endl;
}

void copyTest()
{
    cv::Mat a = cv::Mat::ones(31,31,CV_8UC1);
    cv::cuda::GpuMat src, inte;
    src.upload(a);
    cv::cuda::integral(src, inte);
    cv::cuda::GpuMat dst = cv::cuda::createContinuous(inte.size(), CV_32SC1);
    MSSS msss(5,5);
//    msss.gpuCopyNonContiguous(inte, dst, 2);
    gpuErrChk(cudaMemcpy2D((int32_t*)dst.data, dst.cols * sizeof(int32_t), (int32_t*)inte.data, inte.step, dst.cols * sizeof(int32_t),
                           dst.rows, cudaMemcpyDeviceToDevice));
    inte.download(a);
    std::cout << a << std::endl;
    cv::Mat b;
    dst.download(b);
    std::cout << b << std::endl;
    std::cout << a - b<< std::endl;
}

void MSSSTest(std::string img_path)
{

    //    string img_path = "/home/martin/Pictures/seeker/frame_506.jpg";
        cv::Mat image = cv::imread(img_path, 0);
//        cv::imshow("src", image);
        cv::cuda::GpuMat src = cv::cuda::createContinuous(image.size(), image.type());
//        cv::cuda::GpuMat src;
        src.upload(image);
        auto start = std::chrono::high_resolution_clock::now();
        MSSS msss(image.cols, image.rows);
        std::vector<cv::Rect> bboxes  = msss.detect(src);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "time elapsed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()/1000.0 << std::endl;
        testCUDATime("Time = ", 1, msss.detect(src));
        if(bboxes.size() > 0)
        {
            for(size_t i = 0; i < bboxes.size() ; i++)
            {
                std::cout << bboxes[i] << std::endl;
                cv::rectangle(image, bboxes[i], cv::Scalar(255, 0 , 255), 2, 4);
            }
        }

        cv::imshow("img", image);
        cv::waitKey(0);
}

void MSSS_videoTest(std::string video_path)
{
    cv::VideoCapture cap(video_path);
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    cv::cuda::GpuMat src = cv::cuda::createContinuous(height, width, CV_8UC1);
    MSSS msss(width, height);

    while(true)
    {
        cv::Mat frame ;
        cap.read(frame);
        if(frame.empty())
        {
            std::cout << "End of video" << std::endl;
            break;
        }
        cv::cvtColor(frame, frame, CV_BGRA2GRAY);
        src.upload(frame);
        std::vector<cv::Rect> bboxes = msss.detect(src);    

        for(int i = 0; i < bboxes.size(); i++)
        {
            bboxes[i].y -= 15;
            bboxes[i].height +=15;
            cv::rectangle(frame, bboxes[i], cv::Scalar(255, 0, 255), 2, 4);
        }
        cv::imshow("frame", frame);
        if(cv::waitKey() == 27)
            break;

    }
}
void gray2labtest(std::string img_path)
{
    int size = 18;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::resize(img, img,cv::Size(size, size));
    cv::Mat gray = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    cv::resize(gray, gray,cv::Size(size, size));
    cv::Mat L, a, b, labim;
    MSSS msss(img.cols, img.rows);
    cv::cuda::GpuMat d_img = cv::cuda::createContinuous(img.size(), gray.type());
    cv::cuda::GpuMat d_lab;
    cv::cuda::GpuMat d_L = cv::cuda::createContinuous(gray.size(), CV_8UC1);
    cv::cuda::GpuMat d_a = cv::cuda::createContinuous(gray.size(), CV_8UC1);
    cv::cuda::GpuMat d_b = cv::cuda::createContinuous(gray.size(), CV_8UC1);
    d_img.upload(gray);
    std::cout << "img continuous: " << d_img.isContinuous() << std::endl;
    testCUDATime("Gpu Time = ", 1000, msss.gray2lab(d_img));
    msss.m_L.download(L);
    cv::imshow("GPU", L);
    std::cout << "GPU Result = \n" << L << std::endl;
    d_a.download(a);
    d_b.download(b);

    getExeTime("Cpu time = ", cv::cvtColor(img, labim, CV_RGB2Lab ));
    std::vector<cv::Mat> channels;
    cv::split(labim, channels);
    cv::Mat Lc = channels[0]; // 255;
    cv::Mat ac = channels[1]; // 255;
    cv::Mat bc = channels[2]; // 255;
//    Lc.convertTo(Lc, CV_64F);
//    ac.convertTo(ac, CV_64F);
//    bc.convertTo(bc, CV_64F);
    std::cout << "CPU result = \n" << Lc << std::endl;
    cv::imshow("CPU", Lc);
    cv::waitKey();
}

void integralTest()
{
    cv::Mat a = cv::Mat::ones(3,3, CV_8UC1);
    cv::Mat b;
    cv::integral(a, b);
    std::cout << "CPU: \n" << b << std::endl;

    cv::cuda::GpuMat c;
    cv::cuda::GpuMat d;
    c.upload(a);
    cv::cuda::integral(c, d);
    d.download(b);
    std::cout << "GPU: \n" << b << std::endl;
}
