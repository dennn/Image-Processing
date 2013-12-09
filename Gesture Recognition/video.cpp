#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#define DEFAULT_SAMPLING 0
using namespace std;
using namespace cv;

void getDerivatives (cv::Mat& prevFrame, cv::Mat& currentFrame, cv::Mat& Dx, cv::Mat& Dy,
		cv::Mat& Dt);
void LKTracker (const cv::Mat& Dx, const cv::Mat& Dy, const cv::Mat& Dt, 
		const cv::Point& targetPoint, unsigned windowDimension, cv::Mat& A, 
		cv::Mat& b);
void LK (cv::Mat& prevFrame, cv::Mat& frame);
void print_mat (cv::Mat& m);
void print_mat_d (cv::Mat& m);

int main( int argc, const char** argv )
{
	Mat frame, derivX, derivY, derivT, prevFrame;
	cv::VideoCapture cap;
	int sampling, count;
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
	namedWindow("derivX", 1);
	namedWindow("derivY", 1);
	namedWindow("derivT", 1);

		#ifdef APPLE_WEBCAM
			std::cout << "Waiting for built-in WebCam:" << APPLE_CAM_WAIT << " ms" << std::endl;
			waitKey(APPLE_CAM_WAIT);
		#endif
	count = 0;
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
		cv::cvtColor(frame, frame, CV_BGR2GRAY);

		//If it's the first frame got from the webcam, set the previous frame
		//as an empty frame
		if (i == 0) {
			prevFrame = cv::Mat::zeros(frame.size(), frame.type());
		}
		//Checks whether we should the current frame to get the Derivative images or
		//wait for another one
		if (count >= sampling){
			count = -1;
			getDerivatives (prevFrame, frame, derivX, derivY, derivT);
			imshow("video", frame);
			imshow("derivX", derivX);
			imshow("derivY", derivY);
			imshow("derivT", derivT);
			frame.copyTo(prevFrame);
		}
		count ++;
		// convertScaleAbs( derivT, derivT );
	}



	/******* TESTING CODE
	cv::Mat temp_x = (cv::Mat_<uchar>(4,4) << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
	cv::Mat temp_y = (cv::Mat_<uchar>(4,4) << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
	cv::Mat temp_t = (cv::Mat_<uchar>(4,4) << 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1);
	cv::Mat A = (cv::Mat_<float>(2,2) << 0,0,0,0);
	cv::Mat b = (cv::Mat_<float>(2,1) << 0,0);
	cv::Mat pad_temp_x;
	cv::Mat pad_temp_y;
	cv::Mat pad_temp_t;
	int window_dimension = 2;
	int window_radius = (int)(window_dimension / 2) + 1;

	cv::copyMakeBorder( temp_x, pad_temp_x, 
		window_radius, window_radius, window_radius, window_radius,
		cv::BORDER_REPLICATE );
	cv::copyMakeBorder( temp_y, pad_temp_y, 
		window_radius, window_radius, window_radius, window_radius,
		cv::BORDER_REPLICATE );
	cv::copyMakeBorder( temp_t, pad_temp_t, 
		window_radius, window_radius, window_radius, window_radius,
		cv::BORDER_REPLICATE );


	std::cout << "pad_temp_x\n";
	print_mat(pad_temp_x);
	std::cout << "\n\n\n\n";
	std::cout << "pad_temp_y\n";
	print_mat(pad_temp_y);
	std::cout << "\n\n\n\n";
	std::cout << "pad_temp_t\n";
	print_mat(pad_temp_t);
	std::cout << "\n\n\n\n";
	LKTracker (pad_temp_x,pad_temp_y,pad_temp_t,Point (5,5),window_dimension,A,b);
	std::cout <<"A:\n";
	print_mat_d(A);
	std::cout <<"b:\n";
	print_mat_d(b);
	*****************/

}

//Spatial derivative
void getSpatialDerivative (cv::Mat& currentFrame, cv::Mat& Dx, cv::Mat& Dy){
	static cv::Mat kernelX = (cv::Mat_<float>(3,3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	static cv::Mat kernelY = (cv::Mat_<float>(3,3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
	
	filter2D(currentFrame, Dx, -1 , kernelX, Point(-1,-1), 0, BORDER_DEFAULT );
	// convertScaleAbs( derivX, derivX );
	filter2D(currentFrame, Dy, -1 , kernelY, Point(-1,-1), 0, BORDER_DEFAULT );
	// convertScaleAbs( derivY, derivY );
	cv::normalize(Dx, Dx, 0, 255, CV_MINMAX);
	cv::normalize(Dy, Dy, 0, 255, CV_MINMAX);
}
//Temporal derivative
void getTemporalDerivative (cv::Mat& prevFrame, cv::Mat& currentFrame, cv::Mat& Dt){
	Dt = currentFrame - prevFrame;
	cv::normalize(Dt, Dt, 0, 255, CV_MINMAX);
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
	int sum_x = 0;
	int sum_y = 0;
	//Get Ix(sum_x) first
	for ( int i = windowStart.y ; i < end_row; i++){
		const uchar* current_row = Dx.ptr<uchar>(i);
		current_row+= windowStart.x;
		for (unsigned j = 0; j < windowDimension; j++,current_row++){
			//adds the value from the pixel (windowStart.y+i, windowStart.x+j)
			sum_x += *current_row;
		}
	}

	//Calc sum_y
	for ( int i = windowStart.y ; i < end_row; i++){
		const uchar* current_row = Dy.ptr<uchar>(i);
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
		const cv::Point& targetPoint, unsigned windowDimension, cv::Mat& A, 
		cv::Mat& b){
	//OBS: Dx, Dy and Dt HAVE HAD BEEN PADDED ACCORDINGLY, i. e. WITH THE
	//APPROPRIATE SIZE DEPENDING ON THE DIMENSION OF THE NEIGHBOURHOOD WINDOW

	Point windowStart (targetPoint.x - windowDimension/2, targetPoint.y - windowDimension/2);
	int end_row = windowStart.y + windowDimension;
	int sum_x = 0;
	int sum_y = 0;
	int sum_t = 0;
	//Get Ix(sum_x) first
	for ( int i = windowStart.y ; i < end_row; i++){
		const uchar* current_row = Dx.ptr<uchar>(i);
		current_row+= windowStart.x;
		for (unsigned j = 0; j < windowDimension; j++,current_row++){
			//adds the value from the pixel (windowStart.y+i, windowStart.x+j)
			sum_x += *current_row;
		}
	}

	//Calc sum_y
	for ( int i = windowStart.y ; i < end_row; i++){
		const uchar* current_row = Dy.ptr<uchar>(i);
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

	//Calc sum_t
	for ( int i = windowStart.y ; i < end_row; i++){
		const uchar* current_row = Dt.ptr<uchar>(i);
		current_row+= windowStart.x;
		for (unsigned j = 0; j < windowDimension; j++,current_row++){
			//adds the value from the pixel (windowStart.y+i, windowStart.x+j)
			sum_t += *current_row;
		}
	}
	b.at<float>(0,0) = (-1)*sum_x*sum_t;
	b.at<float>(1,0) = (-1)*sum_y*sum_t;
}
void LK (cv::Mat& prevFrame, cv::Mat& frame){
	cv::Mat derivX, derivY, derivT;
	cv::Mat padDerivX, padDerivY, padDerivT;
	cv::Mat A, AInv, b, v;
	int windowDimension = 2;
	int window_radius = windowDimension / 2;


//	uchar* currentRow;
	cv::Point targetPoint;

	getDerivatives (prevFrame, frame, derivX, derivY, derivT);
	//We need to pad derivX, derivY and derivT
	cv::copyMakeBorder( derivX, padDerivX, 
		window_radius, window_radius, window_radius, window_radius,
		cv::BORDER_REPLICATE );
	cv::copyMakeBorder( derivY, padDerivY, 
		window_radius, window_radius, window_radius, window_radius,
		cv::BORDER_REPLICATE );
	cv::copyMakeBorder( derivT, padDerivT, 
		window_radius, window_radius, window_radius, window_radius,
		cv::BORDER_REPLICATE );


	//Calculates the optical flow for every single pixel in frame
	for (int i = 0; i < frame.rows; i++){
//		currentRow = frame.ptr<uchar>(i);
		for (int j = 0; j < frame.cols; j++){
			targetPoint.x = j+window_radius;
			targetPoint.y = i+window_radius;

			LKTracker (derivX, derivY, derivT, targetPoint, windowDimension, A, b);
			//Now we calculate the optical flow velocity vector v	
			AInv = A.inv();
			v = AInv * b;
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
	cv::Mat detectorResponse (img.size (), CV_32FC1);
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
			uchar e = m.at<uchar>(i,j);
			std::cout << (int)e << ";";
		}
		std::cout << std::endl;
	}
}

