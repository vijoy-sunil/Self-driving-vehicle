/* Signals and signs operations
 * ----------------------------------
 * 1. Traffic lights detection and recognition
 * 2. Yellow Road sign detection
*/ 

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "debug.h"
using namespace cv;
using namespace std;

int processImgR(Mat src);
int processImgG(Mat src);
bool isIntersected(Rect r1, Rect r2);

bool isFirstDetectedR = true;
bool isFirstDetectedG = true;
Rect* lastTrackBoxR;
Rect* lastTrackBoxG;
int lastTrackNumR;
int lastTrackNumG;

int hazard_sign;

int detect_lights(Mat src)
{
	int redCount = 0;
	int greenCount = 0;
	int light_status = 0;

	Mat frame;
	Mat img;
	Mat imgYCrCb;
	Mat imgGreen;
	Mat imgRed;
	Mat src_roi;

	//roi for traffic lights
	Rect roi = Rect(200,0,620,370);	
	src_roi = src(roi);

	// Parameters of brightness
	double a = 0.3; // gain, modify this only.
	double b = (1 - a) * 125; // bias

	src_roi.convertTo(img, img.type(), a, b);
	// Convert to YCrCb color space
	cvtColor(img, imgYCrCb, CV_BGR2YCrCb);

	imgRed.create(imgYCrCb.rows, imgYCrCb.cols, CV_8UC1);
	imgGreen.create(imgYCrCb.rows, imgYCrCb.cols, CV_8UC1);

	// Split three components of YCrCb
	vector<Mat> planes;
	split(imgYCrCb, planes);
	// Traversing to split the color of RED and GREEN according to the Cr component
	MatIterator_<uchar> it_Cr = planes[1].begin<uchar>(),
		it_Cr_end = planes[1].end<uchar>();
	MatIterator_<uchar> it_Red = imgRed.begin<uchar>();
	MatIterator_<uchar> it_Green = imgGreen.begin<uchar>();

	for (; it_Cr != it_Cr_end; ++it_Cr, ++it_Red, ++it_Green)
	{
		// RED, 145<Cr<470 
		if (*it_Cr > 145 && *it_Cr < 470)
			*it_Red = 255;
		else
			*it_Red = 0;

		// GREEN£¬95<Cr<110
		if (*it_Cr > 95 && *it_Cr < 110)
			*it_Green = 255;
		else
			*it_Green = 0;
	}

	//Expansion and corrosion  
	dilate(imgRed, imgRed, Mat(15, 15, CV_8UC1), Point(-1, -1));
	erode(imgRed, imgRed, Mat(1, 1, CV_8UC1), Point(-1, -1));
	dilate(imgGreen, imgGreen, Mat(15, 15, CV_8UC1), Point(-1, -1));
	erode(imgGreen, imgGreen, Mat(1, 1, CV_8UC1), Point(-1, -1));

	redCount = processImgR(imgRed);
	greenCount = processImgG(imgGreen);

	#ifdef LIGHT_DEBUG
		printf("redCount %d greenCount %d\n", redCount, greenCount);
	#endif

	if (redCount > greenCount)
		light_status = 1;
	else if (redCount < greenCount)
		light_status = 0;
	else if(redCount == greenCount)
		light_status = -1;

	return light_status;
}
int processImgR(Mat src)
{
	Mat tmp;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<Point> hull;

	CvPoint2D32f tempNode;
	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* pointSeq = cvCreateSeq(CV_32FC2, sizeof(CvSeq), sizeof(CvPoint2D32f), storage);

	Rect* trackBox;
	Rect* result;
	int resultNum = 0;

	int area = 0;

	src.copyTo(tmp);
	// Extract the contour
	findContours(tmp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	if (contours.size() > 0)
	{
		trackBox = new Rect[contours.size()];
		result = new Rect[contours.size()];

		// Determine the area to track
		for (int i = 0; i < contours.size(); i++)
		{
			cvClearSeq(pointSeq);
			// Get the point set of the convex hull
			convexHull(Mat(contours[i]), hull, true);
			int hullcount = (int)hull.size();
			// Save points of the convex hull
			for (int j = 0; j < hullcount - 1; j++)
			{
				tempNode.x = hull[j].x;
				tempNode.y = hull[j].y;
				cvSeqPush(pointSeq, &tempNode);
			}

			trackBox[i] = cvBoundingRect(pointSeq);
		}

		if (isFirstDetectedR)
		{
			lastTrackBoxR = new Rect[contours.size()];
			for (int i = 0; i < contours.size(); i++)
				lastTrackBoxR[i] = trackBox[i];
			lastTrackNumR = contours.size();
			isFirstDetectedR = false;
		}
		else
		{
			for (int i = 0; i < contours.size(); i++)
			{
				for (int j = 0; j < lastTrackNumR; j++)
				{
					if (isIntersected(trackBox[i], lastTrackBoxR[j]))
					{
						result[resultNum] = trackBox[i];
						break;
					}
				}
				resultNum++;
			}
			delete[] lastTrackBoxR;
			lastTrackBoxR = new Rect[contours.size()];
			for (int i = 0; i < contours.size(); i++)
			{
				lastTrackBoxR[i] = trackBox[i];
			}
			lastTrackNumR = contours.size();
		}

		delete[] trackBox;
	}
	else
	{
		isFirstDetectedR = true;
		result = NULL;
	}
	cvReleaseMemStorage(&storage);

	if (result != NULL)
	{
		for (int i = 0; i < resultNum; i++)
		{
			area += result[i].area();
		}
	}
	delete[] result;

	return area;
}

int processImgG(Mat src)
{
	Mat tmp;

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector< Point > hull;

	CvPoint2D32f tempNode;
	CvMemStorage* storage = cvCreateMemStorage();
	CvSeq* pointSeq = cvCreateSeq(CV_32FC2, sizeof(CvSeq), sizeof(CvPoint2D32f), storage);

	Rect* trackBox;
	Rect* result;
	int resultNum = 0;

	int area = 0;

	src.copyTo(tmp);
	// Extract the contour
	findContours(tmp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	if (contours.size() > 0)
	{
		trackBox = new Rect[contours.size()];
		result = new Rect[contours.size()];

		// Determine the area to track
		for (int i = 0; i < contours.size(); i++)
		{
			cvClearSeq(pointSeq);
			// Get the point set of the convex hull
			convexHull(Mat(contours[i]), hull, true);
			int hullcount = (int)hull.size();
			// Save points of the convex hull
			for (int j = 0; j < hullcount - 1; j++)
			{
				//line(showImg, hull[j + 1], hull[j], Scalar(255, 0, 0), 2, CV_AA);
				tempNode.x = hull[j].x;
				tempNode.y = hull[j].y;
				cvSeqPush(pointSeq, &tempNode);
			}

			trackBox[i] = cvBoundingRect(pointSeq);
		}

		if (isFirstDetectedG)
		{
			lastTrackBoxG = new Rect[contours.size()];
			for (int i = 0; i < contours.size(); i++)
				lastTrackBoxG[i] = trackBox[i];
			lastTrackNumG = contours.size();
			isFirstDetectedG = false;
		}
		else
		{
			for (int i = 0; i < contours.size(); i++)
			{
				for (int j = 0; j < lastTrackNumG; j++)
				{
					if (isIntersected(trackBox[i], lastTrackBoxG[j]))
					{
						result[resultNum] = trackBox[i];
						break;
					}
				}
				resultNum++;
			}
			delete[] lastTrackBoxG;
			lastTrackBoxG = new Rect[contours.size()];
			for (int i = 0; i < contours.size(); i++)
			{
				lastTrackBoxG[i] = trackBox[i];
			}
			lastTrackNumG = contours.size();
		}

		delete[] trackBox;
	}
	else
	{
		isFirstDetectedG = true;
		result = NULL;
	}
	cvReleaseMemStorage(&storage);

	if (result != NULL)
	{
		for (int i = 0; i < resultNum; i++)
		{
			area += result[i].area();
		}
	}
	delete[] result;

	return area;
}


// Determine whether the two rectangular areas are intersected 
bool isIntersected(Rect r1, Rect r2)
{
	int minX = max(r1.x, r2.x);
	int minY = max(r1.y, r2.y);
	int maxX = min(r1.x + r1.width, r2.x + r2.width);
	int maxY = min(r1.y + r1.height, r2.y + r2.height);

	if (minX < maxX && minY < maxY)
		return true;
	else
		return false;
}

int low_t = 150;
int high_t = 255;

Mat detect_sign(Mat src)
{	
	Rect roi = Rect(650,200,630,290);
	Mat roi_src = src(roi);
	Mat im_bin;
	Mat dst = Mat(src.rows, src.cols, CV_8UC3);
	Mat sign = Mat(src.rows, src.cols, CV_8UC3);

	dst = Scalar::all(0);
	sign = Scalar::all(0);
	Mat roi_dst = dst(roi);

	int found_sign = 0;

	double area;
	Mat imgHSV, chn[3];
	cvtColor(roi_src, imgHSV, CV_BGR2HSV);
	split(imgHSV, chn);

	//createTrackbar("min","sign detection", &low_t, 255);

	inRange(chn[1], Scalar(low_t, low_t, low_t), Scalar(high_t, high_t,  high_t), im_bin);	//yellow
	
	

	//morphological opening (remove small objects from the foreground)
	erode(im_bin, im_bin, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
	dilate( im_bin, im_bin, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

	//morphological closing (fill small holes in the foreground)
	dilate( im_bin, im_bin, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
	erode(im_bin, im_bin, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );


	int largest_area=0;
	int largest_contour_index=0;
	Rect bounding_rect;

	vector<vector<Point> > contours; 				// Vector for storing contours
	findContours( im_bin, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE ); // Find the contours in the image

	for( size_t i = 0; i< contours.size(); i++ ) 			// iterate through each contour.
	{
		area = contourArea( contours[i] );  			//  Find the area of contour
		if(area > 150)						//filter out small detections
		{
			if( area > largest_area)		
			{
			    largest_area = area;
			    largest_contour_index = i;               	//Store the index of largest contour
			    bounding_rect = boundingRect(contours[i]); 	// Find the bounding rectangle for biggest contour
			}
			found_sign = 1;
		}
		else
		{
			found_sign = 0;
			hazard_sign = 0;
		}
	}
	if(found_sign == 1)
	{									
		drawContours( roi_dst, contours,largest_contour_index, Scalar( 255, 255, 255 ), -1 ); 	// Draw the largest contour using previously stored index.
		
		bitwise_and(dst,src,sign);
		Mat fin_rect = sign(roi);
		Mat fin_sign = fin_rect(bounding_rect);							//detected sign roi
		rectangle( fin_rect, bounding_rect.tl(), bounding_rect.br(), Scalar(0,255,255), 2, 8, 0 );
		hazard_sign = 1;
	}
		

/*	
	//detect red signs
	Mat mask1, mask2;
	inRange(imgHSV, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
    	inRange(imgHSV, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);
	Mat mask = mask1 | mask2;
	//morphological opening (remove small objects from the foreground)
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
	dilate( mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

	//morphological closing (fill small holes in the foreground)
	dilate( mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
	
	imshow("red", mask);
	int c = cvWaitKey(1);
*/
	return sign;
	
}























