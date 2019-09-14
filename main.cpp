#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat A(300, 400, CV_8UC3, Scalar(0, 255, 0));
    imshow("A", A);
    cout << "my project template . " << endl;
    waitKey(0);
    return 0;
}