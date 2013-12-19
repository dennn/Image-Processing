#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <queue>

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

#define DEFAULT_SAMPLING 0

typedef struct {
	cv::Point start;
	cv::Point end;
	double norm;
} line_segment;

struct compare_line_segments{
	bool operator () (const line_segment& A, const line_segment& B){
		return A.norm < B.norm;
	}
};

void normalizeDisplay(cv::Mat &frame, cv::Mat &frameOut);
void getDerivatives (cv::Mat& prevFrame, cv::Mat& currentFrame, cv::Mat& Dx, cv::Mat& Dy, cv::Mat& Dt);
void LKTracker (const cv::Mat& Dx, const cv::Mat& Dy, const cv::Mat& Dt, const cv::Point& targetPoint, unsigned windowDimension, cv::Mat& A, cv::Mat& b);
void LK (cv::Mat& prevFrame, cv::Mat& frame, priority_queue<line_segment,vector<line_segment>, compare_line_segments>&);
void getWindow (const cv::Mat& frame,cv::Mat& window, cv::Point start, int windowDimension);

int main( int argc, const char** argv )
{
	cv::VideoCapture cap;
	line_segment temp;

	// Decides whether to use a webcam or a video file
	if(argc > 1) {
		cap.open(string(argv[1]));
	} else {
		cap.open(CV_CAP_ANY);
	}

	//checks if the videocapture has been opened
	if(!cap.isOpened()) {
		printf("Error: could not load a camera or video.\n");
	}

	namedWindow("video", 1);

	#ifdef APPLE_WEBCAM
		std::cout << "Waiting for built-in WebCam:" << APPLE_CAM_WAIT << " ms" << std::endl;
		waitKey(APPLE_CAM_WAIT);
	#endif

	Mat frameOriginal, frameResized, prevFrame;
	Mat derivX, derivY, derivT;

	//This queue will store the top N optical flow vectors
	priority_queue<line_segment,vector<line_segment>, compare_line_segments> of_vec_queue;
	for(int i = 0;;i++)
	{
		waitKey(20);
		cap >> frameOriginal;

		if(!frameOriginal.data)
		{
			printf("Error: no frame data.\n");
			break;
		}

		Size s(frameOriginal.size().width/4, frameOriginal.size().height/4);

		cv::resize(frameOriginal, frameResized, s);

		//If it's the first frame got from the webcam, set the previous frame as an empty frame
		if (i == 0) {
			prevFrame = cv::Mat::zeros(frameResized.size(), frameResized.type());
		}

		getDerivatives(prevFrame, frameResized, derivX, derivY, derivT);

		// Run the LK algorithm on our frames
	//	LK (prevFrame, frame_gray, of_vec_queue);
		/*while (of_vec_queue.size() > 0){
			temp = of_vec_queue.top();
			cv::line(frame_large, temp.start, temp.end, Scalar(255,0,0)); 
			of_vec_queue.pop();
		}*/
		imshow("video", frameResized);
		frameResized.copyTo(prevFrame);
	}
}

// Calculate the temporal and spatial derivatives
void getDerivatives (cv::Mat& prevFrame, cv::Mat& currentFrame, cv::Mat& Dx, cv::Mat& Dy,
		cv::Mat& Dt)
{
	Mat frameCopy, prevFrameCopy;

	cv::cvtColor(currentFrame, frameCopy, CV_BGR2GRAY);
	cv::cvtColor(prevFrame, prevFrameCopy, CV_BGR2GRAY);
	frameCopy.convertTo(frameCopy, CV_64F);
	prevFrameCopy.convertTo(prevFrameCopy, CV_64F);

	static cv::Mat kernelX = (cv::Mat_<double>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	static cv::Mat kernelY = (cv::Mat_<double>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);

	//depth is going to be the same as in source. Therefore, CV_8U
	filter2D(frameCopy, Dx, CV_64F , kernelX, Point(-1,-1), 0, BORDER_DEFAULT);
	filter2D(frameCopy, Dy, CV_64F , kernelY, Point(-1,-1), 0, BORDER_DEFAULT);	

	Dt = frameCopy - prevFrameCopy;
	
	Mat Dx_out, Dy_out, Dt_out;

	normalizeDisplay(Dx, Dx_out);
	normalizeDisplay(Dy, Dy_out);
	normalizeDisplay(Dt, Dt_out);

	imshow("derivX", Dx_out);
	imshow("derivY", Dy_out);
	imshow("derivT", Dt_out);
}

void normalizeDisplay(cv::Mat &frame, cv::Mat &frameOut) 
{
	double min, max;
	cv::minMaxLoc(frame, &min, &max);
	double scale = 255.0/(max-min);
	double value = -min * scale;
	frame.convertTo(frameOut, CV_8U, scale, value);
}



void LK (cv::Mat& prevFrame, cv::Mat& frame,
		priority_queue<line_segment,vector<line_segment>, compare_line_segments>& vec_queue)
{
	cv::Mat derivX, derivY, derivT;
	cv::Mat vX, vY;
	cv::Mat A, AInv, b, v;

	A.create(2,2,CV_64F);
	b.create(2,1,CV_64F);
	int windowDimension = 2;
	int window_radius = windowDimension / 2;

	line_segment seg_vec;

	cv::Point windowStart;
	cv::Point windowCenter;
	cv::Point velEndPoint;
	int window_row_size = frame.rows / windowDimension;
	int window_col_size = frame.cols / windowDimension;
	vX.create(window_col_size, window_row_size,CV_64F);
	vY.create(window_col_size, window_row_size,CV_64F);

	getDerivatives(prevFrame, frame, derivX, derivY, derivT);

	imshow("Dx", derivX);
	imshow("Dy", derivY);
	imshow("Dt", derivT);
/*
	cv::Mat window;
	window.create (windowDimension, windowDimension, CV_64F);
	//Calculates the optical flow for every single pixel in frame
	for (int window_row = 0; window_row < window_row_size; window_row++){
		for (int window_col = 0; window_col < window_col_size; window_col++){
			windowStart.x = (window_col*windowDimension);
			windowStart.y = (window_row*windowDimension);

			windowCenter.x = windowStart.x + window_radius;
			windowCenter.y = windowStart.y + window_radius;

			LKTracker (derivX, derivY, derivT, windowStart, windowDimension, A, b);

			double determinants = (A.at<double>(0, 0) * A.at<double>(1, 1)) - (A.at<double>(1, 0) * A.at<double>(0, 1));

			//Now we calculate the optical flow velocity vector v	
			if ( determinant (A) == 0) {
				vX.at<double>(window_row, window_col) = 0;
				vY.at<double>(window_row, window_col) = 0;
				continue;
			}

			invert(A, AInv);
			v = AInv * b;
			vX.at<double>(window_row, window_col) = v.at<double>(0,0);
			vY.at<double>(window_row, window_col) = v.at<double>(0,1);
			seg_vec.norm = norm(v);

			windowCenter.x = windowStart.x + window_radius;
			windowCenter.y = windowStart.y + window_radius;

			velEndPoint.x = (windowCenter.x + v.at<double>(0,0));
			velEndPoint.y = (windowCenter.y + v.at<double>(0,1));
			seg_vec.start = windowCenter;
			seg_vec.end = velEndPoint;

			vec_queue.push(seg_vec);
		}
	}	
	*/
}

void LKTracker (const cv::Mat& Dx, const cv::Mat& Dy, const cv::Mat& Dt, 
		const cv::Point& windowStart, unsigned windowDimension, cv::Mat& A, 
		cv::Mat& b){

	cv::Mat window;
	double sum_x, sum_y, sum_t;


	window.create(windowDimension, windowDimension, CV_64F);

	getWindow(Dx, window, windowStart, windowDimension);
	sum_x = sum(window)[0];

	getWindow(Dy, window, windowStart, windowDimension);
	sum_y = sum(window)[0];

	getWindow(Dt, window, windowStart, windowDimension);
	sum_t = sum(window)[0];

	A.at<double>(0,0) = sum_x*sum_x;
	A.at<double>(0,1) = sum_x*sum_y;
	A.at<double>(1,0) = sum_x*sum_y;
	A.at<double>(1,1) = sum_y*sum_y;

	b.at<double>(0,0) = (-1)*sum_x*sum_t;
	b.at<double>(1,0) = (-1)*sum_y*sum_t;
}

void getWindow (const cv::Mat& frame,cv::Mat& window, cv::Point start, int windowDimension){
	//FIRST COORDINATE IN cv::Mat::at FUNCTION IS THE INDEX OF THE LINE!!!!!!!!!!
	window.create (windowDimension, windowDimension, CV_64F);
	for (int row = 0; row < windowDimension; row++){
		for (int col = 0; col < windowDimension; col++){
			window.at<double>(row,col) = frame.at<double>(row+start.y,col+start.x);
		}
	}
}