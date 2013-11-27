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
void threshold_image (Mat&, int);
/** Global variables */
String logo_cascade_name = "dartcascade.xml";

CascadeClassifier logo_cascade;

string window_name = "Capture - Face detection";

void detect_circle (Mat& frame, Mat& frame_gray, std::vector<Rect>& boards, double threshold){

	vector<Vec3f> circles;
	vector<Rect> candidates;
	double dist;
	double x_c, y_c;
	int min_index = 0;
	double min_dist = 99999.0
;
	Mat grad;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
	//cvtColor( frame, frame_gray, CV_BGR2GRAY );
//	Mat frame = imread(image, CV_GRAY_SCALE);
	Mat grad_x, grad_y;
	  Mat abs_grad_x, abs_grad_y;
		char* window_name = "Gradient image";

	  /// Gradient X
	  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	  Sobel( frame_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_x, abs_grad_x );

	  /// Gradient Y
	  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	  Sobel( frame_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	  convertScaleAbs( grad_y, abs_grad_y );

	  /// Total Gradient (approximate)
	  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

	threshold_image (grad, threshold); 
#ifdef STEP
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );
	imshow(window_name, grad);
	waitKey(0);
#endif
	HoughCircles( grad, circles, CV_HOUGH_GRADIENT, 1, grad.rows/8, 200, 100, 10, 200 );
	
	std::cout << "Classifier: " << boards.size() << "; hough_circle: " <<\
	   	circles.size () << std::endl;
#ifndef FINAL
	  for( size_t i = 0; i < circles.size(); i++ )
  	{
      	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      	int radius = cvRound(circles[i][2]);
      	// circle center
     	 circle( frame, center, 3, Scalar(0,255,0), -1, 8, 0 );
      	// circle outline
      	circle( frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
   	}
#endif
	//	
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
		rectangle(frame, Point(boards[min_index].x-5, boards[min_index].y-5), Point(boards[min_index].x + boards[min_index].width + 5, boards[min_index].y + boards[min_index].height + 5), Scalar( 255, 255, 0 ), 2);
	}
	//imwrite( "output.jpg", frame );

	
}
/** @function main */
int main( int argc, const char** argv )
{
	CvCapture* capture;
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat frame_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

#ifdef GRAY_SCALE
	namedWindow( "Gray scale", CV_WINDOW_AUTOSIZE );
	imshow("Gray scale", frame_gray);
	waitKey(0);
	exit(0);
#endif
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

	GaussianBlur( frame, frame, Size(5, 5), 2, 2 );
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	//equalizeHist( frame_gray, frame_gray );

	//-- Detect faces
	logo_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, 
			Size(50, 50), Size(500,500) );
	//detect_circle (frame_gray, faces, threshold);
	detect_circle (frame,frame_gray, faces, threshold);
#ifndef FINAL
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
#endif

	//-- Save what you got
	imwrite( "output.jpg", frame );
}


void threshold_image(Mat &magnitude, int threshold)
{
    for ( int i = 0; i < magnitude.rows; i++ )
    {
        for( int j = 0; j < magnitude.cols; j++ )
        {
/*
            if (magnitude.at<uchar>( i, j ) > threshold)
                magnitude.at<uchar>( i, j ) = 255;
            else
                magnitude.at<uchar>( i, j ) = 0;
*/

            if (magnitude.at<uchar>( i, j ) < threshold)
                magnitude.at<uchar>( i, j ) = 0;
        }
    }
    
}
