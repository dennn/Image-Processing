#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

void estimateMotion(Mat &dx, Mat &dt, Mat &dy, Mat &matV, int region[], Mat &frameResized, Point centre);
void getDerivatives(Mat &frame, Mat &frame_prev, Mat &dx, Mat &dy, Mat &dt, bool display);
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
	// int maxCorners = 15;
	// double qualityLevel = 0.01;
	// double minDistance = 10;

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
		getDerivatives(frame, prev_frame, dx, dy, dt, true);
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
			matA.at<double>(0, 0) = dx.at<double>(i, j) * dx.at<double>(i, j);
			matA.at<double>(0, 1) = dx.at<double>(i, j) * dy.at<double>(i, j);
			matA.at<double>(1, 0) = dx.at<double>(i, j) * dy.at<double>(i, j);
			matA.at<double>(1, 1) = dy.at<double>(i, j) * dy.at<double>(i, j);
			sumA += matA;

			// Create B for this pixel
			Mat matB;
			matB.create(2, 1, CV_64F);
			matB.at<double>(0, 0) = -1 * dx.at<double>(i, j) * dt.at<double>(i, j);
			matB.at<double>(1, 0) = -1 * dy.at<double>(i, j) * dt.at<double>(i, j);
			sumB += matB;
		}
	}
	// Calculate the vector for this subregion
	if (determinant(sumA) != 0.0) {
		matV = sumA.inv() * sumB;
		double magnitude = sqrt((matV.at<double>(0, 0) * matV.at<double>(0, 0))
													+  matV.at<double>(1, 0) * matV.at<double>(1, 0));
		// cout << magnitude << endl;
		if (magnitude > 1) {
			matV = matV * 30;
			Point vector;
			vector.x = centre.x + matV.at<double>(0, 0);
			vector.y = centre.y + matV.at<double>(1, 0);
			line(frameResized, centre, vector, Scalar(0, 0, 0), 2, 8);
		}
	}
}

//
// Goes over a region and estimates motion in subregions
//
void lkTracker(Mat &dx, Mat &dy, Mat &dt, Mat &frame, Mat &frameResized)
{
	// for (int i = 100; frame.rows)
			Mat matV;
			int step = 10;
			// Loop through the region
			for (int ii = 100; ii < frame.rows-100; ii += step)
			{
				for (int jj = 100; jj < frame.cols-100; jj += step)
				{
					Point centre;
					centre.x = jj + (step / 2);
					centre.y = ii + (step / 2);
					int region[4] = {ii, jj, ii + step, jj + step};
					estimateMotion(dx, dy, dt, matV, region, frameResized, centre);
				}
			}
}

void getDerivatives(Mat &frame, Mat &prev_frame, Mat &dx, Mat &dy, Mat &dt, bool display) {
	// Create derivative matrices of type double
  dx.create(frame.rows - 1, frame.cols - 1, CV_64F);
  dy.create(frame.rows - 1, frame.cols - 1, CV_64F);
  dt.create(frame.rows - 1, frame.cols - 1, CV_64F);

  for(int i = 0; i < frame.rows - 1; i++) {
    for(int j = 0; j < frame.cols - 1; j++) {
    	double iA, iB, iC, iD;

    	// Get dx gradient by calculating difference between two adjacent points
    	// in both frames
    	iA = frame.at<uchar>(i, j + 1) - frame.at<uchar>(i, j);
    	iB = frame.at<uchar>(i + 1, j + 1) - frame.at<uchar>(i + 1, j);
    	iC = prev_frame.at<uchar>(i, j + 1) - prev_frame.at<uchar>(i, j);
    	iD = prev_frame.at<uchar>(i + 1, j + 1) - prev_frame.at<uchar>(i + 1, j);
    	dx.at<double>(i, j) = (iA + iB + iC + iD) / 4;

    	// Get dy gradient by calculating difference between two adjacent points
    	// in both frames
    	iA = frame.at<uchar>(i + 1, j) - frame.at<uchar>(i, j);
    	iB = frame.at<uchar>(i + 1, j + 1) - frame.at<uchar>(i, j + 1);
    	iC = prev_frame.at<uchar>(i + 1, j) - prev_frame.at<uchar>(i, j);
    	iD = prev_frame.at<uchar>(i + 1, j + 1) - prev_frame.at<uchar>(i, j + 1);
    	dy.at<double>(i, j) = (iA + iB + iC + iD) / 4;

    	// Get dt gradient by calculating difference between the same pixels in
    	// each frame
    	iA = frame.at<uchar>(i, j) - prev_frame.at<uchar>(i, j);
    	iB = frame.at<uchar>(i + 1, j) - prev_frame.at<uchar>(i + 1, j);
    	iC = frame.at<uchar>(i, j + 1) - prev_frame.at<uchar>(i, j + 1);
    	iD = frame.at<uchar>(i + 1, j + 1) - prev_frame.at<uchar>(i + 1, j + 1);
    	dt.at<double>(i, j) = (iA + iB + iC + iD) / 4;
    }
  }
  if (display) {
  	// To display we must normalise
	  Mat dxNormalised, dyNormalised, dtNormalised;

	  // Get minimum and maximum for derivatives
	  double dxMax, dxMin, dyMax, dyMin, dtMax, dtMin;
		cv::minMaxLoc(dx, &dxMax, &dxMin);
		cv::minMaxLoc(dy, &dyMax, &dyMin);
		cv::minMaxLoc(dt, &dtMax, &dtMin);
		// Normalise by scaling
		dx.convertTo(dxNormalised, CV_8U, 255.0/(dxMax - dxMin), -dxMin * 255.0/(dxMax - dxMin));
		dy.convertTo(dyNormalised, CV_8U, 255.0/(dyMax - dyMin), -dyMin * 255.0/(dyMax - dyMin));
		dt.convertTo(dtNormalised, CV_8U, 255.0/(dtMax - dtMin), -dtMin * 255.0/(dtMax - dtMin));

		namedWindow("dx", 1);
		namedWindow("dy", 1);
		namedWindow("dt", 1);
	  imshow("dx", dxNormalised);
	  imshow("dy", dyNormalised);
	  imshow("dt", dtNormalised);
	}
}
