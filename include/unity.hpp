#pragma once
#include <opencv2/core.hpp>
#include <vector>
using namespace cv;
using namespace std;

#ifndef _UNITY
#define _UNITY
//show the element of mat(used to test code)
void showMat(Mat RainMat);

//show the element of vector
void showVector(vector<int> index);

//show the element of vector with typt Mat
void showMatVector(vector<Mat> neighbor);

//convert int to string
string Int_String(int index);

#endif