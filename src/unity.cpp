#include "unity.hpp"
//show the element of mat(used to test code)
void showMat(Mat RainMat)
{
   for (int i=0;i<RainMat.rows;i++)
   {
       for (int j=0;j<RainMat.cols;j++)
       {
           cout<<RainMat.at<float>(i,j)<<"  ";
       }
       cout<<endl;
   }
}


//show the element of vector
void showVector(vector<int> index)
{
   for (int i=0;i<index.size();i++)
   {
       cout<<index[i]<<endl;
   }
}

//show the element of vector with typt Mat
void showMatVector(vector<Mat> neighbor)
{
   for (int e=0;e<neighbor.size();e++)
   {
       showMat(neighbor[e]);
   }
}

//convert int to string
string Int_String(int index)
{
    stringstream ss;
    ss << index;
    return ss.str();
}