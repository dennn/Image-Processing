#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void estimateMotion(Mat &dx, Mat &dt, Mat &dy, Mat &matV, int region[], Mat &frameResized, Point centre);
void getDerivatives(Mat &frame, Mat &prev_frame, Mat &dx, Mat &dy, Mat &dt, bool show);
void lkTracker(Mat &dx, Mat &dt, Mat &dy, Mat &frame, Mat &frameResized);

int main(int argc, const char** argv)
{
	cv::VideoCapture cap;
	if(argc > 1)
		cap.open(string(argv[1]));
	else
		cap.open(CV_CAP_ANY);

	if(!cap.isOpened())
		printf("Error: could not load a camera or video.\n");

	Mat frameOriginal, frameResized, frame, prev_frame, dy, dx, dt;
	namedWindow("video", 1);

  	vector<Point2f> corners;
	int maxCorners = 15;
	double qualityLevel = 0.01;
	double minDistance = 10;

	for(int i = 0;;)
	{
		waitKey(20);

		// Get frame from capture
		cap >> frameOriginal;
		if(!frameOriginal.data) {
			printf("Error: no frame data.\n");
			break;
		}

		//Resize the frame
		Size s(640, 480);
		cv::resize(frameOriginal, frameResized, s);

		// Convert frame to gray
		cv::cvtColor(frameResized, frame, CV_BGR2GRAY);

    	// If it's the first frame got from the webcam, set the previous frame
    	// as an empty frame
		if (i == 0) {
			prev_frame = cv::Mat::zeros(frame.size(), frame.type());
			i++;
			continue;
		}

		// Get derivatives
		getDerivatives(frame, prev_frame, dx, dy, dt, false);


		lkTracker(dx, dt, dy, frame, frameResized);

		// Show stuff
		imshow("video", frameResized);

		// Copy current frame to prev_frame
		frame.copyTo(prev_frame);
	}
}

//
// Estimates motion within a given subregion
//
void estimateMotion(Mat &dx, Mat &dt, Mat &dy, Mat &matV, int region[], Mat &frameResized, Point centre)
{
	// Array is formatted like so:
	// 	[ rstart, cstart,
	// 	  rstop, cstop   ]
	int rstart = region[0];
	int rstop = region[2];
	int cstart = region[1];
	int cstop = region[3];

	// Our final A and B matrices that will be used to calculate the vector
	Mat sumA, sumB;
	sumA.create(2, 2, CV_64F);
	sumB.create(2, 1, CV_64F);

	// Loop through the region
	for (int i = rstart; i < rstop; i++)
	{
		for (int j = cstart; j < cstop; j++)
		{
			// Create A for this pixel
			Mat matA;
			matA.create(2, 2, CV_64F);
			matA.at<double>(0, 0) = dx.at<uchar>(i, j) * dx.at<uchar>(i, j);
			matA.at<double>(0, 1) = dx.at<uchar>(i, j) * dy.at<uchar>(i, j);
			matA.at<double>(1, 0) = dx.at<uchar>(i, j) * dy.at<uchar>(i, j);
			matA.at<double>(1, 1) = dy.at<uchar>(i, j) * dy.at<uchar>(i, j);
			sumA += matA;

			// Create B for this pixel
			Mat matB;
			matB.create(2, 1, CV_64F);
			matB.at<double>(0, 0) = -1 * dx.at<uchar>(i, j) * dt.at<uchar>(i, j);
			matB.at<double>(1, 0) = -1 * dy.at<uchar>(i, j) * dt.at<uchar>(i, j);
			sumB += matB;
		}
	}

	// Calculate the vector for this subregion
	if (determinant(sumA) != 0.0) {
		matV = sumA.inv() * sumB;
		// if (abs(matV.at<double>(0, 0)) < 1) {
			//cout << matV << endl;
			// Normalise lol
			matV = matV * 100;
			// cout << matV << endl;
			Point vector;
			vector.x = centre.x + matV.at<double>(0, 0);
			vector.y = centre.y + matV.at<double>(1, 0);
			line(frameResized, centre, vector, Scalar(0, 0, 0), 2, 8);
		// }
	}
}

void lkTracker(Mat &dx, Mat &dt, Mat &dy, Mat &frame, Mat &frameResized)
{
			Mat matV;

			int step = 10;

// Loop through the region
		for (int i = 100; i < frame.rows-100; i += step)
		{
			for (int j = 100; j < frame.cols-100; j += step)
			{
				Point centre;
				centre.x = j + (step / 2);
				centre.y = i + (step / 2);
				int region[4] = {i, j, i + step, j + step};
				estimateMotion(dx, dy, dt, matV, region, frameResized, centre);
			}
		}
}

void getDerivatives(Mat &frame, Mat &prev_frame, Mat &dx, Mat &dy, Mat &dt, bool show)
{
	static cv::Mat kernelX = (cv::Mat_<short>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
  	static cv::Mat kernelY = (cv::Mat_<short>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);

  	// Apply convolution to frame
  	filter2D(frame, dx, CV_8U, kernelX, Point(-1,-1), 0, BORDER_DEFAULT);
  	filter2D(frame, dy, CV_8U, kernelY, Point(-1,-1), 0, BORDER_DEFAULT);

  	// Calculate temporal derivative
  	dt = frame - prev_frame;

  	Mat dx_v, dy_v, dt_v;

  	if (show) {
		cv::normalize(dx, dx_v, 0, 255, CV_MINMAX);
		cv::normalize(dy, dy_v, 0, 255, CV_MINMAX);
		cv::normalize(dt, dt_v, 0, 255, CV_MINMAX);

		namedWindow("dx", 1);
		namedWindow("dy", 1);
		namedWindow("dt", 1);

		imshow("dx", dx_v);
		imshow("dy", dy_v);
		imshow("dt", dt_v);
	}
}


/* Credit to http://creat-tabu.blogspot.co.uk/2013/08/opencv-python-hand-gesture-recognition.html */
void Rect detectHand (Mat &frame) {
	frame = 
}
