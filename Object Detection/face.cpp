/*  Dartboard detector using OpenCV
 *  Image Processing and Computer Vision 2013
 *  aa1462, ff13400, do1303
 */

#include "opencv.hpp"
#include "objdetect/objdetect.hpp"
#include "highgui/highgui.hpp"
#include "imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndSave(Mat frame);
void circle_hough_transform (Mat& img, vector<Vec3f>& circles);
void detectCircles(Mat& frame, Mat& frame_gray, std::vector<Rect>& boards);
void circleHough(Mat& img_gray, vector<Vec3f>& circles);
void detectLines(Mat &frame, vector<Vec4i>lines);

/** Global variables */
CascadeClassifier dartClassifier;
string dartHaarFile = "dartcascade.xml";

string window_name = "Capture - Face detection";

/** 
 * @function The main function is responsible for loading in the image and calling the other feature
 * detecting functions. It then saves out the image.
 */
int main( int argc, const char** argv )
{
    //Read in the image file as both a color and a grayscale image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat frame_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    
    /* This is for debugging and can be set in the MakeFile. It defines whether to show the
     * gray scale version of the image
     */
#ifdef GRAY_SCALE
	namedWindow("Gray scale", CV_WINDOW_AUTOSIZE);
	imshow("Gray scale", frame_gray);
	waitKey(0);
#endif
    
    //If we can't load the Haar feature file then quit
	if(!dartClassifier.load(dartHaarFile)){
        printf("--(!)Error loading\n");
        return -1;
    };
    
    //Detect the dartboards before saving the file
	detectAndSave(frame);
    
	return 0;
}

/** 
 * @function detectAndSave - This function creates different copies of the images based on what
 *                           they require. It then calls the relevant functions.
 *
 * @param img - A reference to the image that will be checked for dartboards
 * @param threshold - A value used
 *
 */
void detectAndSave(Mat img)
{
	vector<Rect> dartboards;
    vector<Vec4i> lines;
	Mat img_gray;
	Mat blurred_img, blurred_img_gray;
	Mat equalized_img_gray, equalized_blurred_img_gray;
    
    //Smooth the image with a gaussian blur
	GaussianBlur(img, blurred_img, Size(5, 5), 2, 2);
    //Create a copy of the image that is in grayscale
	cvtColor(img, img_gray, CV_BGR2GRAY);
    //Create a copy of the smoothed image in grayscale
	cvtColor(blurred_img, blurred_img_gray, CV_BGR2GRAY);
    //Equalize the histograms because converting it to grayscale shifts the colour range
	equalizeHist(blurred_img_gray, equalized_blurred_img_gray);
	equalizeHist(img_gray, equalized_img_gray);
    
	//EQUALIZED
    //	dartClassifier.detectMultiScale( equalized_img_gray, dartboards, 1.1, 4, 0|CV_HAAR_SCALE_IMAGE,
    //			Size(50, 50), Size(500,500) );
    
	//BLURRED
	dartClassifier.detectMultiScale(blurred_img_gray, dartboards, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE,
                                  Size(50, 50), Size(500,500) );
	//BLURRED + EQUALIZED
	//logo_cascade.detectMultiScale( equalized_blurred_img_gray, dartboards, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE,
	//		Size(50, 50), Size(500,500) );
    
	//Detect circles without blur
    //	detect_circle (img,img_gray, dartboards);
    
	//Detect circles with blur
	detectCircles(img, blurred_img_gray, dartboards);
    
    //Detect the lines
    detectLines(img, lines);
    
#ifndef FINAL
	for( int i = 0; i < dartboards.size(); i++ )
	{
		rectangle(img, Point(dartboards[i].x, dartboards[i].y), Point(dartboards[i].x + dartboards[i].width, dartboards[i].y + dartboards[i].height), Scalar( 0, 255, 0 ), 2);
	}
#endif
    
    //Write out the final image
	imwrite("output.jpg", img);
}


/**
 * @function detectCircles - Given a gray frame, this function draws all circles detected.
 *
 * @param frame - The image to draw the detected circles onto
 * @param frameGray - A grayscale version of the image
 * @param boards - A vector of all the dartboards that have been detected with the Haar feature classifier
 *
 */
void detectCircles(Mat& frame, Mat& frameGray, std::vector<Rect>& boards)
{
	vector<Rect> candidates;
	vector<Vec3f> circles;
	double dist;
	double x_c, y_c;
	int min_index = 0;
	double min_dist = 99999.0;
    
    circleHough(frameGray, circles);
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

	for (int i = 0; i < circles.size (); i++){
		for (int j = 0; j < boards.size (); j++){
			x_c = boards[j].x + (boards[j].width/2.0);
			y_c = boards[j].y + (boards[j].height/2.0);
			Point rect_c (x_c, y_c);
			Point circ_c (circles[i][0], circles[i][1]);
			dist = norm(rect_c - circ_c);
			if ( dist < min_dist) {
				min_dist = dist;
				min_index = j;
			}
		}
	}
	if ( boards.size () > 0 ){
		rectangle(frame, Point(boards[min_index].x-5, boards[min_index].y-5), Point(boards[min_index].x + boards[min_index].width + 5, boards[min_index].y + boards[min_index].height + 5), Scalar( 255, 255, 0 ), 2);
	}
}

/**
 * @function circleHough - Given a gray frame, this function applies a sobel filter to it before executing the 
 *                         Hough circles function.
 *
 * @param imgGray - The image to draw the detected circles onto
 * @param circles - A vector where all detected circles will be stored
 *
 */
void circleHough(Mat& imgGray, vector<Vec3f>& circles)
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Mat grad;
  	int ddepth = CV_16S;
  	int scale = 1;
  	int delta = 0;

	// Gradient X
	Sobel(imgGray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs( grad_x, abs_grad_x );

	// Gradient Y
	Sobel(imgGray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
#ifdef STEP
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
	imshow(window_name, grad);
	waitKey(0);
#endif
	HoughCircles(grad, circles, CV_HOUGH_GRADIENT, 1, imgGray.rows/3, 150, 80, 10, 200);
}

void detectLines(Mat &frame, vector<Vec4i>lines)
{
    Mat frameGray, thresholdedImage, detectedLines;
    
    //Convert to grayscale
    cvtColor(frame, frameGray, CV_BGR2GRAY);
    
    //Apply a binary threshold
    threshold(frameGray, thresholdedImage, 220, 255, CV_THRESH_TRUNC);
    
    //Apply Canny edge detection
    Canny(thresholdedImage, thresholdedImage, 50, 200, 3);
    
    //Apply Hough Lines detection
    HoughLinesP(thresholdedImage, lines, 1, CV_PI/180, 40, 30, 10);
    for(size_t i = 0; i < lines.size(); i++)
    {
        Vec4i l = lines[i];
        line(frame, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,255,255), 1, CV_AA);
    }
}