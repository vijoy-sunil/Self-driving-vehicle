/* Traffic operations
 * ----------------------------------
 * 1. Vehicle detection
 * 2. Vehicle distance meaurement
 * 3. Pedestrian detection
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

int hazard_people,hazard_vehicle;

String cascadeName = "./cascades/cars.xml"; // Trained cascade for detecting cars.

#define CAMERA_FOCAL_LENGTH 100	// 100mm
#define CAMERA_HEIGHT 1300	// 1300mm

#define MIN_DISTANCE 25

Mat detect_vehicle(Mat src)
{

	//variables to write text in window
	int font = FONT_HERSHEY_SIMPLEX;
	double fontScale = 0.6;
	int thickness = 2;
	char unit[] = " m";
	char finalText[25];

	// variables to calculate vehicle distance
	uint16_t Yb = 0;
	uint16_t Yh = 0;
	float d = 0;

	Rect roi = Rect(310,360,520,250);						
	//Mat detect_roi = src(roi);
	Mat vehicle_detection_img = src(roi);

	Mat vehicle_distance_img;
	src.copyTo(vehicle_distance_img);

	//ROI below the horizon, where we want vehicles to be detected
	//Mat vehicle_detection_img = src( Rect( 0, (src.rows/2), src.cols, (src.rows - (src.rows/2)) ) );

	CascadeClassifier haar_cascade_cars;
	haar_cascade_cars.load(cascadeName);
	vector< Rect_<int> > cars;
	haar_cascade_cars.detectMultiScale(vehicle_detection_img, cars, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(60, 60) );

	for(int i = 0; i < cars.size(); i++)
	{
		Rect car_i = cars[i];
		//Draw rectangle/bounding box
		rectangle(vehicle_distance_img, Point(car_i.x+310,car_i.y+360), Point(car_i.x + 310 + car_i.width, car_i.y + 360 + car_i.height), CV_RGB(255,0,0), 3); 

		//calculate vehicle distance
		Yb = ((car_i.y+(360)-10) + car_i.height); // bottom-line coordinate
		Yh = (360); // horizon coordinate
		d = ((CAMERA_FOCAL_LENGTH * CAMERA_HEIGHT)/(Yb - Yh))*16/(100*9); // AR of video = 16:9

		if (d < MIN_DISTANCE)
			hazard_vehicle = 1;
		else
			hazard_vehicle = 0;

		// Write text
		Point textOrg((car_i.x + 310), car_i.y+(360)-4);
		sprintf(finalText,"%3.0f%s",d,unit);
		putText(vehicle_distance_img, finalText, textOrg, font, fontScale, CV_RGB(0, 255 , 255), thickness,8);
	}
 
	return vehicle_distance_img;
}


Mat detect_people(const HOGDescriptor &hog, Mat img)
{
	vector<Rect> found, found_filtered;
	
	Rect roi = Rect(0,300,1280,275);
	Mat img_roi = img(roi);
	Mat dst = Mat (img.rows, img.cols, CV_8UC3);
	dst = Scalar::all(0);

	hazard_people = 0;

	// Run the detector with default parameters. to get a higher hit-rate
	// (and more false alarms, respectively), decrease the hitThreshold and
	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).

	hog.detectMultiScale(img_roi, found, 0, Size(8,8), Size(16,16), 1.05, 2);
	for(size_t i = 0; i < found.size(); i++ )
	{
	Rect r = found[i];

	size_t j;
	// Do not add small detections inside a bigger detection.
	for ( j = 0; j < found.size(); j++ )
	    if ( j != i && (r & found[j]) == r )
		break;

	if ( j == found.size() )
	    found_filtered.push_back(r);
	}

	for (size_t i = 0; i < found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];

		// The HOG detector returns slightly larger rectangles than the real objects,
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		rectangle(img_roi, r.tl(), r.br(), cv::Scalar(255,0,0), 3);

		hazard_people = 1;		//set hazard_people flag
	}

	return dst;
}

