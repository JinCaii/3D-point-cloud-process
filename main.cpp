#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(void)
{
    cv::Mat img = cv::imread("C:/Users/User/Desktop/111.jpg");
    cv::namedWindow("img", cv::WINDOW_NORMAL);
    cv::imshow("img", img);
    cv::resizeWindow("img", 800, 600);
    cv::waitKey(0);

    return 0;
}
