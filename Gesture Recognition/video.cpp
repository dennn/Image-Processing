#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main( int argc, const char** argv )
{
	cv::VideoCapture cap;
	if(argc > 1)
	{
		cap.open(string(argv[1]));
	}
	else
	{
		cap.open(CV_CAP_ANY);
	}
	if(!cap.isOpened())
	{
		printf("Error: could not load a camera or video.\n");
	}
	Mat frame, kernelX, kernelY, derivX, derivY, derivT, prevFrame;
	kernelX = (cv::Mat_<double>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	kernelY = (cv::Mat_<double>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	namedWindow("video", 1);
	namedWindow("derivX", 1);
	namedWindow("derivY", 1);
	namedWindow("derivT", 1);
	for(int i = 0;;i++)
	{
		waitKey(20);
		cap >> frame;
		cv::cvtColor(frame, frame, CV_BGR2GRAY);
		if (i == 0) {
			prevFrame = cv::Mat::zeros(frame.size(), frame.type());
		}
		filter2D(frame, derivX, -1 , kernelX, Point(-1,-1), 0, BORDER_DEFAULT );
		// convertScaleAbs( derivX, derivX );
		filter2D(frame, derivY, -1 , kernelY, Point(-1,-1), 0, BORDER_DEFAULT );
		// convertScaleAbs( derivY, derivY );
		derivT = frame - prevFrame;
		// convertScaleAbs( derivT, derivT );
		if(!frame.data)
		{
			printf("Error: no frame data.\n");
			break;
		}
		cv::normalize(derivX, derivX, 0, 255, CV_MINMAX);
		cv::normalize(derivY, derivY, 0, 255, CV_MINMAX);
		cv::normalize(derivT, derivT, 0, 255, CV_MINMAX);
		imshow("video", frame);
		imshow("derivX", derivX);
		imshow("derivY", derivY);
		imshow("derivT", derivT);
		frame.copyTo(prevFrame);
	}
}
