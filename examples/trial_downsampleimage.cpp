#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <opencv2/opencv.hpp>

using namespace std;

int main(int argc, char* argv[])
{
    cv::Mat img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    vector<cv::Mat> img_vec(3);

    cv::Mat black = cv::Mat::zeros(img.rows, img.cols, img.type());
    img_vec.at(0) = black;
    img_vec.at(1) = black;
    img_vec.at(2) = img;

    cv::Mat color;
    cv::merge(img_vec, color);
    cv::imshow("display", color);
    cv::waitKey(0);
}
