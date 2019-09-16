#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
    Mat a = imread("a.jpg");
    
    imshow("ss", a);
    waitKey();
    return 0;
}