#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <queue>

#include <iostream>
#include <stdio.h>
#define DEFAULT_SAMPLING 0
#define OF_VEC_NUM 9999999999
using namespace std;
using namespace cv;

typedef struct {
	cv::Point start;
	cv::Point end;
	float norm;
} line_segment;

struct compare_line_segments{
	bool operator () (const line_segment& A, const line_segment& B){
		return A.norm < B.norm;
	}
};


void calcMatA (const cv::Mat& Dx, const cv::Mat& Dy, const cv::Point& targetPoint,
		unsigned windowDimension, cv::Mat& A);
void getSpatialDerivative (cv::Mat& currentFrame, cv::Mat& Dx, cv::Mat& Dy);
void getDerivatives (cv::Mat& prevFrame, cv::Mat& currentFrame, cv::Mat& Dx, cv::Mat& Dy,
		cv::Mat& Dt);
void LKTracker (const cv::Mat& Dx, const cv::Mat& Dy, const cv::Mat& Dt, 
		const cv::Point& targetPoint, unsigned windowDimension, cv::Mat& A, 
		cv::Mat& b);
void LK (cv::Mat& prevFrame, cv::Mat& frame,
		priority_queue<line_segment,vector<line_segment>, compare_line_segments>&, int);
void getWindow (const cv::Mat& frame,cv::Mat& window, cv::Point start, int windowDimension);
void print_mat (cv::Mat& m);
void print_mat_d (cv::Mat& m);

int main( int argc, const char** argv )
{


	Mat frame, frame_gray, derivX, derivY, derivT, prevFrame;
	cv::VideoCapture cap;
	int sampling, count;
	line_segment temp;
	if(argc > 1)
	{
		cap.open(string(argv[1]));
		if (argc > 2){
			sampling = atoi(argv[2]);
		}
		else
			sampling = DEFAULT_SAMPLING;
	}
	else
	{
		cap.open(CV_CAP_ANY);
	}

	//checks if the videocapture has been opened
	if(!cap.isOpened())
	{
		printf("Error: could not load a camera or video.\n");
	}
	namedWindow("video", 1);
	namedWindow("vX", 1);
	namedWindow("vY", 1);
	namedWindow("derivT", 1);

		#ifdef APPLE_WEBCAM
			std::cout << "Waiting for built-in WebCam:" << APPLE_CAM_WAIT << " ms" << std::endl;
			waitKey(APPLE_CAM_WAIT);
		#endif
	count = 0;


	//This queue will store the top N optical flow vectors
	priority_queue<line_segment,vector<line_segment>, compare_line_segments> of_vec_queue;
	for(int i = 0;;i++)
	{
		waitKey(20);
		cap >> frame;
		if(!frame.data)
		{
			printf("Error: no frame data.\n");
			break;
		}
		//converts the frame extracted to gray scale
		cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);

		//If it's the first frame got from the webcam, set the previous frame
		//as an empty frame
		if (i == 0) {
			prevFrame = cv::Mat::zeros(frame_gray.size(), frame_gray.type());
		}

		//Checks whether we should the current frame to get the Derivative images or
		//wait for another one
		LK (prevFrame, frame_gray, of_vec_queue, OF_VEC_NUM);
		while (of_vec_queue.size() > 0){
			temp = of_vec_queue.top();
			cv::line(frame, temp.start, temp.end, Scalar(255,0,0)); 
			of_vec_queue.pop();
		}
		imshow("video", frame);
		frame_gray.copyTo(prevFrame);
	}
}

//Spatial derivative
void getSpatialDerivative (cv::Mat& currentFrame, cv::Mat& Dx, cv::Mat& Dy){
	static cv::Mat kernelX = (cv::Mat_<short>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	static cv::Mat kernelY = (cv::Mat_<short>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);


	//depth is going to be the same as in source. Therefore, CV_8U
	filter2D(currentFrame, Dx, CV_32F , kernelX, Point(-1,-1), 0, BORDER_DEFAULT );
	filter2D(currentFrame, Dy, CV_32F , kernelY, Point(-1,-1), 0, BORDER_DEFAULT );

	//cv::normalize(Dx, Dx, 0, 255, CV_MINMAX);
	//cv::normalize(Dy, Dy, 0, 255, CV_MINMAX);
}
//Temporal derivative
void getTemporalDerivative (cv::Mat& prevFrame, cv::Mat& currentFrame, cv::Mat& Dt){
	currentFrame.convertTo(currentFrame, CV_32F);
	prevFrame.convertTo(prevFrame, CV_32F);
	Dt = currentFrame - prevFrame;
	//cv::normalize(Dt, Dt, 0, 255, CV_MINMAX);
}
void getDerivatives (cv::Mat& prevFrame, cv::Mat& currentFrame, cv::Mat& Dx, cv::Mat& Dy,
		cv::Mat& Dt){
	getSpatialDerivative 		(currentFrame, Dx, Dy);
	getTemporalDerivative 	(prevFrame, currentFrame, Dt);
}


void calcMatA (const cv::Mat& Dx, const cv::Mat& Dy, const cv::Point& targetPoint,
		unsigned windowDimension, cv::Mat& A){
	Point windowStart (targetPoint.x - windowDimension/2, targetPoint.y - windowDimension/2);
	int end_row = windowStart.y + windowDimension;
	float sum_x = 0;
	float sum_y = 0;
	float sum_t = 0;
	//Get Ix(sum_x) first
	for ( int i = windowStart.y ; i < end_row; i++){
		const float* current_row = Dx.ptr<float>(i);
		current_row+= windowStart.x;
		for (unsigned j = 0; j < windowDimension; j++,current_row++){
			//adds the value from the pixel (windowStart.y+i, windowStart.x+j)
			sum_x += *current_row;
		}
	}

	//Calc sum_y
	for ( int i = windowStart.y ; i < end_row; i++){
		const float* current_row = Dy.ptr<float>(i);
		current_row+= windowStart.x;
		for (unsigned j = 0; j < windowDimension; j++,current_row++){
			//adds the value from the pixel (windowStart.y+i, windowStart.x+j)
			sum_y += *current_row;
		}
	}
	A.at<float>(0,0) = sum_x*sum_x;
	A.at<float>(0,1) = sum_x*sum_y;
	A.at<float>(1,0) = sum_x*sum_y;
	A.at<float>(1,1) = sum_y*sum_y;

}
void LKTracker (const cv::Mat& Dx, const cv::Mat& Dy, const cv::Mat& Dt, 
		const cv::Point& windowStart, unsigned windowDimension, cv::Mat& A, 
		cv::Mat& b){

	cv::Mat window;
	float sum_x, sum_y, sum_t;


	window.create(windowDimension, windowDimension, CV_32F);

	getWindow(Dx, window, windowStart, windowDimension);
	sum_x = sum(window)[0];
	std::cout << "window_x:" << window << std::endl;
	std::cout << "sum_x:" << sum_x << std::endl;
	
	getWindow(Dy, window, windowStart, windowDimension);
	sum_y = sum(window)[0];
	std::cout << "window_y:" << window << std::endl;
	std::cout << "sum_y:" << sum_y << std::endl;
	getWindow(Dt, window, windowStart, windowDimension);
	sum_t = sum(window)[0];
	std::cout << "window_t:" << window << std::endl;
	std::cout << "sum_t:" << sum_t << std::endl;

	A.at<float>(0,0) = sum_x*sum_x;
	A.at<float>(0,1) = sum_x*sum_y;
	A.at<float>(1,0) = sum_x*sum_y;
	A.at<float>(1,1) = sum_y*sum_y;

	b.at<float>(0,0) = (-1)*sum_x*sum_t;
	b.at<float>(1,0) = (-1)*sum_y*sum_t;
}
void LK (cv::Mat& prevFrame, cv::Mat& frame,
		priority_queue<line_segment,vector<line_segment>, compare_line_segments>& vec_queue, 
		int max_vec_num){



	cv::Mat derivX, derivY, derivT;
	cv::Mat padDerivX, padDerivY, padDerivT;
	cv::Mat vX, vY;
	cv::Mat A, AInv, b, v;
	int vec_num = 0;


	A.create(2,2,CV_32F);
	b.create(2,1,CV_32F);
	int windowDimension = 2;
	int window_radius = windowDimension / 2;

	line_segment seg_vec;

	cv::Point windowStart;
	cv::Point windowCenter;
	cv::Point velEndPoint;
	int window_i_size = frame.rows / windowDimension;
	int window_j_size = frame.cols / windowDimension;
	vX.create(window_j_size, window_i_size,CV_32F);
	vY.create(window_j_size, window_i_size,CV_32F);

	getDerivatives (prevFrame, frame, derivX, derivY, derivT);
	//We need to pad derivX, derivY and derivT

	cv::Mat window;
	window.create (windowDimension, windowDimension, CV_32F);
	//Calculates the optical flow for every single pixel in frame
	for (int window_i = 0; window_i < window_i_size; window_i++){
		for (int window_j = 0; window_j < window_j_size; window_j++){
			windowStart.x = (window_j*windowDimension);
			windowStart.y = (window_i*windowDimension);

			windowCenter.x = windowStart.x + window_radius;
			windowCenter.y = windowStart.y + window_radius;
			cv::Mat x, y, t;

			getWindow (frame, x, Point(5,0), 5);
			getWindow (frame, y, Point(7,0), 5);
			getWindow (frame, t, Point(0,0), 5);

			std::cout << "Dx:" << x << std::endl;
			std::cout << "Dy:" << y << std::endl;
			std::cout << "Dt:" << t << std::endl;
			
			LKTracker (x,y,t,Point(3,3),2, A, b);
			std::cout << "A:" << A << std::endl;
			std::cout << "b:" << b << std::endl;

			exit (0);

			/*
			LKTracker (derivX, derivY, derivT, windowStart, windowDimension, A, b);

			//Now we calculate the optical flow velocity vector v	
			if ( determinant (A) == 0) {
				vX.at<float>(window_i, window_j) = 0;
				vY.at<float>(window_i, window_j) = 0;
				continue;
			}
			invert(A, AInv);

			v = AInv * b;
			//Draw v onto frame
			//original image, base point of the velocity vector, end of the velocity vector

			seg_vec.norm = norm(v);
			vX.at<float>(window_i, window_j) = v.at<float>(0,0);
			vY.at<float>(window_i, window_j) = v.at<float>(0,1);
			std::cout << "A:" << A << std::endl;
			std::cout << "InvA:" << AInv << std::endl;
			std::cout << "b:" << b << std::endl;

			std::cout << "v:" << v << std::endl;

			windowCenter.x = windowStart.x + window_radius;
			windowCenter.y = windowStart.y + window_radius;

			velEndPoint.x = (windowCenter.x + v.at<float>(0,0));
			velEndPoint.y = (windowCenter.y + v.at<float>(0,1));
			seg_vec.start = windowCenter;
			seg_vec.end = velEndPoint;

			vec_queue.push(seg_vec);
			*/
		}
	}	
	std::cout << "\n\n\n\n\nNEW FRAME\n" ;
	exit(0);

	imshow("vX", vX);
	imshow("vY", vY);
}


void getWindow (const cv::Mat& frame,cv::Mat& window, cv::Point start, int windowDimension){
	//FIRST COORDINATE IN cv::Mat::at FUNCTION IS THE INDEX OF THE LINE!!!!!!!!!!
	window.create (windowDimension, windowDimension, CV_32F);
	for (int row = 0; row < windowDimension; row++){
		for (int col = 0; col < windowDimension; col++){
			window.at<float>(row,col) = frame.at<float>(row+start.y,col+start.x);
		}
	}
}

/////////HARRIS CORNER DETECTOR/////////////////////////


float harris_corner_evaluator (cv::Mat& A, float k){
	//This function calculates the cornerness of a given pixel window
	///////////////// _				_
	//The matrix A = |Ix*Ix		Ix*Iy|
	//////////////// |Iy*Ix		Iy*Iy|

	float det_A;
	float trace_A;
	float* row_0 = A.ptr<float>(0);
	float* row_1 = A.ptr<float>(1);

	//Calculates the determinant of A
	det_A = row_0[0]*row_1[1] - (row_0[1]*row_1[0]);
	//Calculates the trace of A
	trace_A = row_0[0] + row_1[1];

	return det_A - k*(trace_A*trace_A);
}

void harris_corner_detector (cv::Mat& img){
	cv::Mat Dx, Dy, A;
	cv::Mat detectorResponse (img.size (), CV_32F);
	float* resp_row;
	int windowDimension = 2;
	Point targetPixel (0,0);

	getSpatialDerivative (img, Dx, Dy);
	//iterate over the entire image
	for (int i = 0; i < img.rows; i++){
		//gets a new line
		resp_row = detectorResponse.ptr<float>(i);
		for (int j = 0; j < img.cols; j++){
			targetPixel.x = j; //column index
			targetPixel.y = i; //Row index
			//WE NEED TO PAD  Dx AND Dy FIRST!!!				
			calcMatA (Dx, Dy, targetPixel, windowDimension, A); 
			resp_row[j] = harris_corner_evaluator (A, 0.15);

		}
	}
		
}
//////////////////////////////////////////////////////////////////////









void print_mat_d (cv::Mat& m){
	for (int i = 0; i < m.rows; i++){
		for (int j = 0; j < m.cols; j++){
			std::cout << m.at<float>(i,j) << ";";
		}
		std::cout << std::endl;
	}
}
void print_mat (cv::Mat& m){
	for (int i = 0; i < m.rows; i++){
		for (int j = 0; j < m.cols; j++){
			float e = m.at<float>(i,j);
			std::cout << (int)e << ";";
		}
		std::cout << std::endl;
	}
}

