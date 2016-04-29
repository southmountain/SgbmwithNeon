/*The parameters of function ComputeDisparity.
image1:the left image(input). 
	Compared to using GRAY image , using RGB image makes the quality better but the time consumes more.   
image2:the right image(input).
disparity:the disparity map(output).
	Yon should normalize it before showing it in a window for that its type is CV_16S.If you just want to get the true disparity map for calculating the depth map,you should divide it by DISP_SCALE(Its value 		is 16 in sgbm.cpp) and then get the absolute number,e.g.,abs(disparity/16).
mindis:the minmal value of disparity(input).
	It should be negative number.Because in this program,we assume that point(x,y) in the left image corresponds point(x-d,y) in the right image,where x means the column,y means the row and d means the value 		of disparity.  
numdis:The range of disparity(input).
	You should insure numdis % 16 == 0.
numthread:the number of threads(input).Default value is 1.
*/
#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;
void ComputeDisparity(Mat &image1,Mat &image2,Mat &disparity,int mindis,int numdis,int numthread=1);

int main(){
	Mat img1 = imread("tsukuba1color.png", 0);
	Mat img2 = imread("tsukuba2color.png", 0);
	Mat disparity,s;
	for(int i=1;i<20;i++){
	int64 t1=getTickCount();
	ComputeDisparity(img1,img2,disparity,-16,16);
	t1=getTickCount()-t1;
	cout<<"1 threads:"<<t1/getTickFrequency()<<endl;
	int64 t2=getTickCount();
	ComputeDisparity(img1,img2,disparity,-16,16,2);
	t2=getTickCount()-t2;
	cout<<"2 threads:"<<t2/getTickFrequency()<<endl;
	int64 t3=getTickCount();
	ComputeDisparity(img1,img2,disparity,-16,16,3);
	t3=getTickCount()-t3;
	cout<<"3 threads:"<<t3/getTickFrequency()<<endl;
	int64 t4=getTickCount();
	ComputeDisparity(img1,img2,disparity,-16,16,4);
	t4=getTickCount()-t4;
	cout<<"4 threads:"<<t4/getTickFrequency()<<endl;
	int64 t5=getTickCount();
	ComputeDisparity(img1,img2,disparity,-16,16,5);
	t5=getTickCount()-t5;
	cout<<"5 threads:"<<t5/getTickFrequency()<<endl;
	}
	normalize(disparity, s, 0, 255, CV_MINMAX, CV_8U);
	namedWindow("disparity",1);	
	imshow("disparity", s);
	waitKey();
	return 0;
}
