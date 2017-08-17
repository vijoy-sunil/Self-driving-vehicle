/* to do 
 * vehicle distance
 * sift traffic sign
 * change vehicle detect to mat implementation
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

#include "lane_ops.h"
#include "signal_ops.h"
#include "traffic_ops.h"
#include "debug.h"

void help(void)
{
	printf("\n\n....................HELP.......................\n");
	printf(" ./auto-pilot <VIDEO-NAME.FORMAT> --feature <feature_num> --store <OUTPUT VIDEO FILENAME>\n\n");
	printf(".................Feature_num options...........\n");
	printf(" 0. All features enable\n");
	printf(" 1. Lane detection\n");
	printf(" 2. Lane departure warning\n");
	printf(" 3. Vehicle detection + distance estimation\n");
	printf(" 4. Pedestrian detection\n");
	printf(" 5. Traffic lights recognition\n");
	printf(" 6. Road sign recogntion\n\n");

	printf("..................EXAMPLE usage..............\n");
	printf(" ./auto-pilot front.AVI --feature 6 --store out.avi\n");
	printf(" Output video is stored in output directory\n\n");
	printf(" Press ECS to quit OR p/P to pause and r/R to resume\n");
	printf(".............................................\n\n");
}


VideoCapture capture;
extern int hazard_people, hazard_vehicle;

int slider = 0;	
void onTrackbarSlide(int , void*)								//seek bar call back function
{
	capture.set(CV_CAP_PROP_POS_FRAMES,slider);
}

int main( int argc, char* argv[] )
{
	char input_file[30], output_file[30], output_dir[30];
	int fillR, fillG, fillB, feature_num,c;
	int light_status = 0, lane_change = 0, all_feature = 0;
	//hazard flag
	int hazard_lane = 0, hazard_light = 0;
	

	if (argc == 6)
	{
		strcpy(input_file,argv[1]);
		sscanf(argv[3], "%d", &feature_num);
		strcpy(output_file,argv[5]);		
	}
	else
	{
		help();
		return -1;
	}

	capture = VideoCapture(input_file);							//read source video

	int ROWS = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int COLS = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	double frames = capture.get(CV_CAP_PROP_FRAME_COUNT);

	Mat im_frame,im_gray, im_lane_debug, hazard_icon, im_merge1, im_merge2, im_merge3;
	Mat im_hough = Mat(ROWS,COLS,CV_8UC3);
	Mat im_vehicle = Mat(ROWS,COLS,CV_8UC3);
	Mat im_final = Mat(ROWS, COLS, CV_8UC3);
	Mat im_lane = Mat(ROWS,COLS,CV_8UC3);
	Mat im_haar = Mat(ROWS, COLS, CV_8UC3);
	Mat im_hog = Mat(ROWS, COLS, CV_8UC3);
	Mat im_light_status = Mat(ROWS, COLS, CV_8UC3);
	Mat im_lights = Mat(ROWS, COLS, CV_8UC3);
	Mat im_people = Mat(ROWS, COLS, CV_8UC3);
	Mat im_hazard = Mat(ROWS, COLS, CV_8UC3);
	Mat im_hazard_detect = Mat(ROWS, COLS, CV_8UC3);

	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	
	sprintf(output_dir,"./output/%s",output_file);
	VideoWriter video(output_dir,CV_FOURCC('M','J','P','G'),30, Size(COLS,ROWS),true);

	cvNamedWindow("source video",CV_WINDOW_NORMAL);
	createTrackbar("Seek","source video", &slider, frames,onTrackbarSlide);

	while(1)
	{
		capture.read(im_frame);
		if(im_frame.empty())									//stop after last frame 
		{
		    printf("End of video\n");
		    break;
		}
		setTrackbarPos("Seek", "source video", slider + 1);
		imshow("source video",im_frame);					
		c = cvWaitKey(1);

		switch(feature_num)									//turn feature on/off
		{

			case 0: //all feature 
				//bitwise_or everything
				all_feature = 1;
				bitwise_or(im_lane,im_vehicle,im_merge1);
				bitwise_or(im_merge1,im_lights,im_merge2);
				bitwise_or(im_merge2,im_people,im_merge3);
				bitwise_or(im_merge3,im_hazard_detect,im_final);

				cvNamedWindow("final video",CV_WINDOW_NORMAL);
				imshow("final video",im_final);					
				c = cvWaitKey(1);


			case 1: //lane_detection
				im_hough = detect_lane(im_frame);
				bitwise_or(im_hough,im_frame,im_lane);

				#ifdef LANE_DEBUG
					im_lane_debug = im_frame;
					rectangle(im_lane_debug,Point(600,460),Point(630,490),Scalar(0,255,0),3);
					rectangle(im_lane_debug,Point(600,520),Point(630,550),Scalar(0,255,0),3);
					cvNamedWindow("lane threshold debug",CV_WINDOW_NORMAL);
					imshow("lane threshold debug",im_lane_debug);					
					c = cvWaitKey(1);
				#endif	

				if(all_feature == 0) {
					cvNamedWindow("lane detection",CV_WINDOW_NORMAL);
					imshow("lane detection",im_lane);					
					c = cvWaitKey(1);
					break;	
				}	

			case 2: //lane_warning
				lane_change = detect_lane_change(im_frame);
				if (lane_change == 1)							//set hazard flag on lane change
					hazard_lane = 1;
				else
					hazard_lane = 0;

				if(all_feature == 0) {
					goto hazard_detect;
					break;	
				}
	
			case 3: //vehicle detection
				im_haar = detect_vehicle(im_frame);					
				bitwise_or(im_haar,im_frame,im_vehicle);
	
				if(all_feature == 0) {
					cvNamedWindow("vehicle detection",CV_WINDOW_NORMAL);
					imshow("vehicle detection",im_vehicle);					
					c = cvWaitKey(1);
					goto hazard_detect;
					break;	
				}
	
			case 4: //pedestrian detection
				im_hog = detect_people(hog,im_frame);
				bitwise_or(im_hog,im_frame,im_people);

				if(all_feature == 0) {
					cvNamedWindow("pedestrian detection",CV_WINDOW_NORMAL);
					imshow("pedestrian detection",im_people);					
					c = cvWaitKey(1);
					goto hazard_detect;
					break;	
				}
	
			case 5: //traffic lights recognition
				light_status = detect_lights(im_frame);
				//draw light status on blank image im_light_status
				if (light_status == 1) {
					fillR = 255; 
					fillG = 0; 
					fillB = 0; }
				else if (light_status == 0) {
					fillR = 0; 
					fillG = 255; 
					fillB = 0; }
				else if (light_status == -1) {
					fillR = 255; 
					fillG = 255; 
					fillB = 255; }
				putText(im_light_status, "Traffic Light", cvPoint(1090,535), FONT_HERSHEY_SIMPLEX, 0.6, cvScalar(255,255,255), 1, CV_AA);
				rectangle(im_light_status,Point(1100,550),Point(1200,650),Scalar(fillB,fillG,fillR),-1);

				bitwise_or(im_light_status,im_frame,im_lights);

				#ifdef LIGHT_DEBUG
					printf("Traffic Light Status (red 1, green 0) : %d\n",light_status);
				#endif	

				if( light_status == 1)							//set hazard flag on red light
					hazard_light = 1;
				else
					hazard_light = 0;

				if(all_feature == 0) {
					cvNamedWindow("traffic light recognition",CV_WINDOW_NORMAL);
					//rectangle(im_lights,Point(200,0),Point(820,370),Scalar(255,255,255),3);
					imshow("traffic light recognition",im_lights);					
					c = cvWaitKey(1);
					goto hazard_detect;
					break;	
				}

			case 6: //sign recognition
				if(all_feature == 0)
				{
					break;
				}

			case 7: //hazard detection
				hazard_detect:
				hazard_icon = imread("input/hazard.png");
				if((hazard_lane == 1) || (hazard_people == 1) || (hazard_light == 1) || (hazard_vehicle == 1))	
				{
					hazard_icon.copyTo(im_hazard(Rect(1100,200,hazard_icon.rows,hazard_icon.cols)));
					if(hazard_light == 1)
						putText(im_hazard, "T", cvPoint(1120, 195), FONT_HERSHEY_SIMPLEX, 0.6, cvScalar(255,255,255), 1, CV_AA);
					if(hazard_lane == 1)
						putText(im_hazard, "L", cvPoint(1135, 195), FONT_HERSHEY_SIMPLEX, 0.6, cvScalar(255,255,255), 1, CV_AA);
					if(hazard_people == 1)
						putText(im_hazard, "P", cvPoint(1150, 195), FONT_HERSHEY_SIMPLEX, 0.6, cvScalar(255,255,255), 1, CV_AA);
					if(hazard_vehicle == 1)
						putText(im_hazard, "V", cvPoint(1165, 195), FONT_HERSHEY_SIMPLEX, 0.6, cvScalar(255,255,255), 1, CV_AA);
					
				}
				else
					im_hazard = Scalar::all(0);
				
				bitwise_or(im_hazard, im_frame, im_hazard_detect);
				
				if(all_feature == 0) {
					cvNamedWindow("hazard detection",CV_WINDOW_NORMAL);
					imshow("hazard detection",im_hazard_detect);					
					c = cvWaitKey(1);
				}
				break;	
			

			default:
				printf("Invalid input feature\n");
				break;
	
		}


		if (c == 27)
			break;
		if (c == 112 || c ==80)									//pause/resume video
		{
			printf("Video Paused\n");
			while(1)
			{
				if(feature_num == 0)
				{
					imshow("source video",im_frame);					
					c = cvWaitKey(1);	
					imshow("final video",im_final);
					c = cvWaitKey(1);
				}

				if (c == 114 || c==82)
					break;
			}
			printf("Video Resumed\n");
		}

		video.write(im_final);									//write video
	}

	capture.release();
	destroyAllWindows();
};




