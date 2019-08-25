#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

using namespace cv;

int main() {
	Mat blue(640, 480, CV_8UC3, Scalar(255, 0, 0));
	namedWindow("1", WINDOW_AUTOSIZE);
	imshow("1", blue);
	waitKey(0);
	return 0;
}
