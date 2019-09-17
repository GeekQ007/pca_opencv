#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include <map>
#include <sstream>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

const double u = 0.01f;
const double v = 0.01f;   //the global parameter
const int MNeighbor = 40; //the M nearest neighbors
// Number of components to keep for the PCA
const int num_components = 20;
//the M neighbor mats
vector<Mat> MneighborMat;
//the class index of M neighbor mats
vector<int> MneighborIndex;
//the number of object which used to training
const int Training_ObjectNum = 5;
//the number of image that each object used
const int Training_ImageNum = 7;
//the number of object used to testing
const int Test_ObjectNum = 40;
//the image number
const int Test_ImageNum = 3;
// Normalizes a given image into a value range between 0 and 255.
Mat norm_0_255(const Mat &src)
{
    // Create and return normalized image:
    Mat dst;
    switch (src.channels())
    {
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat> &src, int rtype, double alpha = 1, double beta = 0)
{
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if (n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for (int i = 0; i < n; i++)
    {
        //
        if (src[i].empty())
        {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            cout << error_message << endl;
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if (src[i].total() != d)
        {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            cout << error_message << endl;
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if (src[i].isContinuous())
        {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
        else
        {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

//convert int to string
string Int_String(int index)
{
    stringstream ss;
    ss << index;
    return ss.str();
}

////show the element of mat(used to test code)
//void showMat(Mat RainMat)
//{
//    for (int i=0;i<RainMat.rows;i++)
//    {
//        for (int j=0;j<RainMat.cols;j++)
//        {
//            cout<<RainMat.at<float>(i,j)<<"  ";
//        }
//        cout<<endl;
//    }
//}

//
////show the element of vector
//void showVector(vector<int> index)
//{
//    for (int i=0;i<index.size();i++)
//    {
//        cout<<index[i]<<endl;
//    }
//}
//

//void showMatVector(vector<Mat> neighbor)
//{
//    for (int e=0;e<neighbor.size();e++)
//    {
//        showMat(neighbor[e]);
//    }
//}
//Training function
void Trainging()
{
    // Holds some training images:
    vector<Mat> db;
    // This is the path to where I stored the images, yours is different!
    for (int i = 1; i <= Training_ObjectNum; i++)
    {
        for (int j = 1; j <= Training_ImageNum; j++)
        {
            string filename = "s" + Int_String(i) + "/" + Int_String(j) + ".jpg";
            db.push_back(imread(filename, IMREAD_GRAYSCALE));
        }
    }
    // Build a matrix with the observations in row:
    Mat data = asRowMatrix(db, CV_32FC1);
    // Perform a PCA:
    PCA pca(data, Mat(), 0, num_components);
    // And copy the PCA results:
    Mat mean = pca.mean.clone();
    Mat eigenvalues = pca.eigenvalues.clone();
    Mat eigenvectors = pca.eigenvectors.clone();
    // The mean face:
    //imshow("avg", norm_0_255(mean.reshape(1, db[0].rows)));
    // The first three eigenfaces:
    //imshow("pc1", norm_0_255(pca.eigenvectors.row(0)).reshape(1, db[0].rows));
    //imshow("pc2", norm_0_255(pca.eigenvectors.row(1)).reshape(1, db[0].rows));
    //imshow("pc3", norm_0_255(pca.eigenvectors.row(2)).reshape(1, db[0].rows));
    ////get and save the training image information which decreased on dimensionality
    Mat mat_trans_eigen;
    Mat temp_data = data.clone();
    Mat temp_eigenvector = pca.eigenvectors.clone();
    gemm(temp_data, temp_eigenvector, 1, NULL, 0, mat_trans_eigen, GEMM_2_T);
    //save the eigenvectors
    FileStorage fs("./eigenvector.xml", FileStorage::WRITE);
    fs << "eigenvector" << eigenvectors;
    fs << "TrainingSamples" << mat_trans_eigen;
    fs.release();
}
//Line combination of test sample used by training samples
//parameter:y stand for the test sample column vector;
//x stand for the training samples matrix
Mat LineCombination(Mat y, Mat x)
{
    //the number of training samples
    size_t col = x.cols;
    //the result mat
    Mat result = Mat(col, 1, CV_32FC1);
    //the transposition of x and also work as a temp matrix
    Mat trans_x_mat = Mat(col, col, CV_32FC1);
    //construct the identity matrix
    Mat I = Mat::ones(col, col, CV_32FC1);
    //solve the Y=XA
    //result=x.inv(DECOMP_SVD);
    //result*=y;
    Mat temp = (x.t() * x + u * I);
    Mat temp_one = temp.inv(DECOMP_SVD);
    Mat temp_two = x.t() * y;
    result = temp_one * temp_two;
    return result;
}
//Error test
//parameter:y stand for the test sample column vector;
//x stand for the training samples matrix
//coeff stand for the coefficient of training samples
void ErrorTest(Mat y, Mat x, Mat coeff)
{
    //the array store the coefficient
    map<double, int> Efficient;
    //compute the error
    for (int i = 0; i < x.cols; i++)
    {
        Mat temp = x.col(i);
        double coefficient = coeff.at<float>(i, 0);
        temp = coefficient * temp;
        double e = norm((y - temp), NORM_L2);
        Efficient[e] = i; //insert a new element
    }
    //select the minimum w col as the w nearest neighbors
    map<double, int>::const_iterator map_it = Efficient.begin();
    int num = 0;
    //the map could sorted by the key one
    while (map_it != Efficient.end() && num < MNeighbor)
    {
        MneighborMat.push_back(x.col(map_it->second));
        MneighborIndex.push_back(map_it->second);
        ++map_it;
        ++num;
    }
    //return MneighborMat;
}
//error test of two step
//parameter:MneighborMat store the class information of M nearest neighbor samples
int ErrorTest_Two(Mat y, Mat x, Mat coeff)
{
    int result;
    bool flag = true;
    double minimumerror;
    //
    map<int, vector<Mat>> ErrorResult;
    //count the class of M neighbor
    for (int i = 0; i < x.cols; i++)
    {
        //compare
        //Mat temp=x.col(i)==MneighborMat[i];
        //showMat(temp);
        //if (temp.at<float>(0,0)==255)
        //{
        int classinf = MneighborIndex[i];
        double coefficient = coeff.at<float>(i, 0);
        Mat temp = x.col(i);
        temp = coefficient * temp;
        ErrorResult[classinf / Training_ImageNum].push_back(temp);
        //}
    }
    //
    map<int, vector<Mat>>::const_iterator map_it = ErrorResult.begin();
    while (map_it != ErrorResult.end())
    {
        vector<Mat> temp_mat = map_it->second;
        int num = temp_mat.size();
        Mat temp_one;
        temp_one = Mat::zeros(temp_mat[0].rows, temp_mat[0].cols, CV_32FC1);
        while (num > 0)
        {
            temp_one += temp_mat[num - 1];
            num--;
        }
        double e = norm((y - temp_one), NORM_L2);
        if (flag)
        {
            minimumerror = e;
            result = map_it->first + 1;
            flag = false;
        }
        if (e < minimumerror)
        {
            minimumerror = e;
            result = map_it->first + 1;
        }
        ++map_it;
    }
    return result;
}
//testing function
//parameter:y stand for the test sample column vector;
//x stand for the training samples matrix
int testing(Mat x, Mat y)
{
    // the class that test sample belongs to
    int classNum;
    //the first step: get the M nearest neighbors
    Mat coffecient = LineCombination(y.t(), x.t());
    //cout<<"the first step coffecient"<<endl;
    //showMat(coffecient);
    //map<Mat,int> MneighborMat=ErrorTest(y,x,coffecient);
    ErrorTest(y.t(), x.t(), coffecient);
    //cout<<"the M neighbor index"<<endl;
    //showVector(MneighborIndex);
    //cout<<"the M neighbor mats"<<endl;
    //showMatVector(MneighborMat);
    //the second step:
    //construct the W nearest neighbors mat
    int row = x.cols; //should be careful
    Mat temp(row, MNeighbor, CV_32FC1);
    for (int i = 0; i < MneighborMat.size(); i++)
    {
        Mat temp_x = temp.col(i);
        if (MneighborMat[i].isContinuous())
        {
            MneighborMat[i].convertTo(temp_x, CV_32FC1, 1, 0);
        }
        else
        {
            MneighborMat[i].clone().convertTo(temp_x, CV_32FC1, 1, 0);
        }
    }
    //cout<<"the second step mat"<<endl;
    //showMat(temp);
    Mat coffecient_two = LineCombination(y.t(), temp);
    //cout<<"the second step coffecient"<<endl;
    //showMat(coffecient_two);
    classNum = ErrorTest_Two(y.t(), temp, coffecient_two);
    return classNum;
}

int main(int argc, const char *argv[])
{
    //the number which test true
    int TrueNum = 0;
    //the Total sample which be tested
    int TotalNum = Test_ObjectNum * Test_ImageNum;
    //if there is the eigenvector.xml, it means we have got the training data and go to the testing stage directly;
    FileStorage fs("eigenvector.xml", FileStorage::READ);
    if (fs.isOpened())
    {
        //if the eigenvector.xml file exist,read the mat data
        Mat mat_eigenvector;
        fs["eigenvector"] >> mat_eigenvector;
        Mat mat_Training;
        fs["TrainingSamples"] >> mat_Training;
        for (int i = 1; i <= Test_ObjectNum; i++)
        {
            int ClassTestNum = 0;
            for (int j = Training_ImageNum + 1; j <= Training_ImageNum + Test_ImageNum; j++)
            {
                string filename = "s" + Int_String(i) + "/" + Int_String(j) + ".jpg";
                Mat TestSample = imread(filename, IMREAD_GRAYSCALE);
                Mat TestSample_Row;

                TestSample.reshape(1, 1).convertTo(TestSample_Row, CV_32FC1, 1, 0); //convert to row mat

                Mat De_deminsion_test;
                gemm(TestSample_Row, mat_eigenvector, 1, NULL, 0, De_deminsion_test, GEMM_2_T); // get the test sample which decrease the dimensionality
                //cout<<"the test sample"<<endl;
                //showMat(De_deminsion_test.t());
                //cout<<"the training samples"<<endl;
                //showMat(mat_Training);
                int result = testing(mat_Training, De_deminsion_test);
                //cout<<"the result is"<<result<<endl;
                if (result == i)
                {
                    TrueNum++;
                    ClassTestNum++;
                }
                MneighborIndex.clear();
                MneighborMat.clear(); //及时清除空间
            }
            cout << "第" << Int_String(i) << "类测试正确的图片数:  " << Int_String(ClassTestNum) << endl;
        }
        fs.release();
    }
    else

    {
        Trainging();
    }
    // Show the images:
    //waitKey(0);
    // Success!
    return 0;
}