#include "opencv.hpp"
#include "objdetect/objdetect.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndSave( Mat frame, double );

/** Global variables */
String logo_cascade_name = "dartcascade.xml";

CascadeClassifier logo_cascade;

string window_name = "Capture - Face detection";

void detect_circle (Mat& frame, std::vector<Rect>& boards, double threshold){

	vector<Vec3f> circles;
	vector<Rect> candidates;
	Mat frame_gray;// = frame;
	double dist;
	double x_c, y_c;
	int min_index = 0;
	double min_dist = 99999.0;
	//cvtColor( frame, frame_gray, CV_BGR2GRAY );
//	Mat frame = imread(image, CV_GRAY_SCALE);
	HoughCircles( frame, circles, CV_HOUGH_GRADIENT, 1, frame.rows/8, 200, 100, 0, 0 );

	for (int i = 0; i < circles.size (); i++){
		for (int j = 0; j < boards.size (); j++){
//			dist = pow((circles[i][0] - boards[j][0]), 2) 
			x_c = boards[j].x + (boards[j].width/2.0);
			y_c = boards[j].y + (boards[j].height/2.0);
			Point rect_c (x_c, y_c);
			Point circ_c (circles[i][0], circles[i][1]);		
			dist = norm(rect_c - circ_c);
			if ( dist < min_dist){
				min_dist = dist;
				min_index = j;
			}
		}
	}
	
	if ( boards.size () > 0 ){
		rectangle(frame, Point(boards[min_index].x, boards[min_index].y), Point(boards[min_index].x + boards[min_index].width, boards[min_index].y + boards[min_index].height), Scalar( 0, 255, 0 ), 2);
	}
	imwrite( "output.jpg", frame );

	
}
/** @function main */
int main( int argc, const char** argv )
{
	CvCapture* capture;
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	//Mat frame = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	GaussianBlur( frame, frame, Size(9, 9), 2, 2 );

	//detect_circle (frame);
	//-- 1. Load the cascades
	//logo_cascade.load(logo_cascade_name);
	if( !logo_cascade.load( logo_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	detectAndSave( frame, atof(argv[2]) );

	return 0;
}

/** @function detectAndSave */
void detectAndSave( Mat frame, double threshold )
{
	std::vector<Rect> faces;
	Mat frame_gray;// = frame;

	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	logo_cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	std::cout << faces.size() << std::endl;
	detect_circle (frame_gray, faces, threshold);
	//for( int i = 0; i < faces.size(); i++ )
//	{
//		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
//	}


	//-- Save what you got
//	imwrite( "output.jpg", frame );

}
