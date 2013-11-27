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
void Hist_and_Backproj(int, void* );

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
//
//
//
//
//
//
//
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
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );
	//imshow(window_name, grad);
	//waitKey(0);

	HoughCircles( grad, circles, CV_HOUGH_GRADIENT, 1, frame.rows/10, 200, 100, 10, 200 );
	
	  for( size_t i = 0; i < circles.size(); i++ )
  	{
      	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      	int radius = cvRound(circles[i][2]);
      	// circle center
     	 circle( frame, center, 3, Scalar(0,255,0), -1, 8, 0 );
      	// circle outline
      	circle( frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
   	}
	
/*
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
*/	
//	if ( boards.size () > 0 ){
//		rectangle(frame, Point(boards[min_index].x, boards[min_index].y), Point(boards[min_index].x + boards[min_index].width, boards[min_index].y + boards[min_index].height), Scalar( 0, 255, 0 ), 2);
//	}
	//imwrite( "output.jpg", frame );

	
}

Mat src; Mat hsv; Mat hue; Mat test; Mat hsv_test; Mat hue_test;
int bins = 25;
/** @function main */
int main( int argc, const char** argv )
{
	// CvCapture* capture;
	// Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	// //Mat frame = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	// //detect_circle (frame);
	// //-- 1. Load the cascades
	// //logo_cascade.load(logo_cascade_name);
	// if( !logo_cascade.load( logo_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// detectAndSave( frame, atof(argv[2]) );

	/// Read the image
	src = imread( argv[1], 1 );
	test = imread(argv[2], 1);
	/// Transform it to HSV
	cvtColor( src, hsv, CV_BGR2HSV );
	cvtColor( test, hsv_test, CV_BGR2HSV );

	/// Use only the Hue value
	hue.create( hsv.size(), hsv.depth() );
	int ch[] = { 0, 0 };
	mixChannels( &hsv, 1, &hue, 1, ch, 1 );

	hue_test.create( hsv_test.size(), hsv_test.depth() );
	mixChannels( &hsv_test, 1, &hue_test, 1, ch, 1 );

	/// Create Trackbar to enter the number of bins
	char* window_image = "Source image";
	namedWindow( window_image, CV_WINDOW_AUTOSIZE );
	createTrackbar("* Hue  bins: ", window_image, &bins, 180, Hist_and_Backproj );
	Hist_and_Backproj(0, 0);

	/// Show the image
	imshow( window_image, src );

	/// Wait until user exits the program
	waitKey(0);

	return 0;
}

/**
 * @function Hist_and_Backproj
 * @brief Callback to Trackbar
 */
void Hist_and_Backproj(int, void* )
{
  MatND hist;
  int histSize = MAX( bins, 2 );
  float hue_range[] = { 0, 180 };
  const float* ranges = { hue_range };

  /// Get the Histogram and normalize it
  calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false );
  normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

  /// Get Backprojection
  MatND backproj;
  calcBackProject( &hue_test, 1, 0, hist, backproj, &ranges, 1, true );

  /// Draw the backproj
  imshow( "BackProj", backproj );

  /// Draw the histogram
  int w = 400; int h = 400;
  int bin_w = cvRound( (double) w / histSize );
  Mat histImg = Mat::zeros( w, h, CV_8UC3 );

  for( int i = 0; i < bins; i ++ )
     { rectangle( histImg, Point( i*bin_w, h ), Point( (i+1)*bin_w, h - cvRound( hist.at<float>(i)*h/255.0 ) ), Scalar( 0, 0, 255 ), -1 ); }

  imshow( "Histogram", histImg );
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
	logo_cascade.detectMultiScale( frame_gray, faces, 1.1, 3, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
	std::cout << faces.size() << std::endl;
	//detect_circle (frame_gray, faces, threshold);
	detect_circle (frame,frame_gray, faces, threshold);
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}


	//-- Save what you got
	imwrite( "output.jpg", frame );

}
