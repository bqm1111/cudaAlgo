#include <iostream>
#include "MSSS.hpp"
#include "test.hpp"
using namespace std;

int main(int argc, char* argv[])
{
    std::string img_path = "/home/martin/Pictures/seeker/frame_5163.jpg";
//    test();
//    thrustTest();
//    copyTest();
//    MSSSTest(argv[1]);
    MSSS_videoTest(argv[1]);
//    gray2labtest(img_path);
//    integralTest();
}
