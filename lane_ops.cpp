/* Lane operations
 * ----------------------------------
 * 1. Lane detection
 * 2. Lane departure warning
*/ 

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

#include "debug.h"

int low = 0, high = 255;
Mat detect_lane(Mat src)
{

	Mat dst, im_white, gray;
	cvtColor(src, gray, CV_BGR2GRAY);
	
	Mat cdst = Mat(gray.rows,gray.cols,CV_8UC3);
	Mat masked_image= Mat(gray.rows,gray.cols,CV_8UC1);
	Mat mask = Mat(gray.rows,gray.cols,CV_8UC1);
	mask = Scalar::all(0);
	
	vector< Point> contour1;								//roi where lane detection is applied
	vector< Point> contour2;

	contour1.push_back(Point(500, 470));							//left lane
	contour1.push_back(Point(550, 470));
	contour1.push_back(Point(500, 565));
	contour1.push_back(Point(350, 565));

	contour2.push_back(Point(660, 470));							//right lane
	contour2.push_back(Point(710, 470));
	contour2.push_back(Point(850, 565));
	contour2.push_back(Point(740, 565));

	const Point *pts1 = (const cv::Point*) Mat(contour1).data;
	const Point *pts2 = (const cv::Point*) Mat(contour2).data;

	fillConvexPoly(mask, pts1, 4, cv::Scalar(255, 255, 255));				//polygon masking
	fillConvexPoly(mask, pts2, 4, cv::Scalar(255, 255, 255));

	bitwise_and(gray,mask,masked_image);

	Rect roi_h = Rect(600,460,30,30);							//adaptive thresholding						
	Mat mean_thresh_high = gray(roi_h);

	Rect roi_l = Rect(600,520,30,30);						
	Mat mean_thresh_low = gray(roi_l);

	int sum_high = 0, avg_high = 0;
	int sum_low = 0, avg_low = 0;
	for(int i = 0; i< mean_thresh_high.rows; i++)
	{
		for(int j = 0; j< mean_thresh_high.cols; j++)
		{
			sum_high += mean_thresh_high.at<uchar>(i,j);
			sum_low += mean_thresh_low.at<uchar>(i,j);
		}
	}

	int new_thresh = 99;
	avg_high = sum_high/(mean_thresh_high.rows * mean_thresh_high.cols);
	avg_low = sum_low/(mean_thresh_low.rows * mean_thresh_low.cols);

	
	if(avg_high >= 80)
		new_thresh = 110;
	else if(avg_high >=70 && avg_high<80)
		new_thresh = 99;
	else if(avg_high >=60 && avg_high <70)
		new_thresh = 89;
	else if(avg_high>=50 && avg_high<60)
		new_thresh = 79;
	else if(avg_high >=30 && avg_high <50)
		new_thresh = 70;

	if((avg_high - avg_low > 20) || (avg_high - avg_low < -10))
		new_thresh = 120;
	

	#ifdef LANE_DEBUG
		printf("avg_high %d\t avg_low %d\t DIFF: %d\t",avg_high,avg_low,avg_high-avg_low); 
		printf("new thresh %d\n",new_thresh);
	#endif

	low = new_thresh;

	inRange(masked_image, Scalar(low, low, low), Scalar(high, high, high), im_white);	//thresholding for white lanes

	//morphological opening (remove small objects from the foreground)
	//erode(im_white, im_white, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
	//dilate( im_white, im_white, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

	//morphological closing (fill small holes in the foreground)
	//dilate( im_white, im_white, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
	//erode(im_white, im_white, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

	Canny(im_white, dst, 25, 100, 3);

	cdst = Scalar::all(0);	

	vector<Vec4i> lines;
	
	HoughLinesP(dst, lines, 1, CV_PI/180, 25, 10, 80 );				//threshold-minlength-maxgapbtwpts

	for( size_t i = 0; i < lines.size(); i++ )
	{
		Vec4i l = lines[i];
		Point a = Point(l[0], l[1]);	
		Point b= Point(l[2], l[3]);

		if((b.y - a.y) < -12 || (b.y-a.y) > 12)					//remove horizontal lines
		{	
			line( cdst, Point(l[0], l[1]), Point(l[2],l[3]), Scalar(0,255,0), 3, CV_AA);	
		}

	}

	return cdst;
}


int detect_lane_change(Mat src)
{
	Mat dst ;
	int lane_change  = 0;

	cvtColor(src, dst, CV_BGR2GRAY);

	Rect roi_up = Rect(600,460,30,30);						//roi to detect lane change
	Rect roi_down = Rect(600,520,30,30);
	Rect roi_left = Rect(570,490,30,30);						//roi to detect lane change
	Rect roi_right = Rect(630,490,30,30);

	Mat mean_up = dst(roi_up);
	Mat mean_down = dst(roi_down);
	Mat mean_left = dst(roi_left);
	Mat mean_right = dst(roi_right);

	int sum_up = 0, sum_down = 0, sum_left = 0, sum_right = 0;
	int avg_up = 0, avg_down = 0, avg_left = 0, avg_right = 0;

	for (int i = 0; i <mean_up.rows; i++)						//calculate avg intensity in the two rois
	{
		for (int j = 0; j<mean_up.cols; j++)
		{
			sum_up += mean_up.at<uchar>(i,j);
			sum_down += mean_down.at<uchar>(i,j);
			sum_left += mean_left.at<uchar>(i,j);
			sum_right += mean_right.at<uchar>(i,j);
		}
	}
		
	avg_up = sum_up/(mean_up.rows * mean_up.cols);
	avg_down = sum_down/(mean_down.rows * mean_down.cols);
	avg_left = sum_left/(mean_left.rows * mean_left.cols);
	avg_right = sum_right/(mean_right.rows * mean_right.cols);

	float vertical = avg_up - avg_down;
	float horizontal = avg_left - avg_right;

	//printf("vert %f \t hori %f\n",vertical, horizontal);
	
	if(vertical - horizontal > 25)
		lane_change = 1;
	else 
		lane_change = 0;	
	
	return lane_change;
}

