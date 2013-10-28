#include <iostream>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>
#include <math.h>

#define M_PI 3.141592654

/** 
 * Function to add gaussian blur to an image
 *
 * Requires: An input image to be blured
 *           An output image that will contain the blurred input
 *           A kernel size
 */
void Sobel(
	cv::Mat &input);

int main ( int argc, char ** argv )
{
	/* Load up all of the images of car numberplates */
	cv::Mat coin1 = cv::imread("coins1.png", CV_LOAD_IMAGE_GRAYSCALE);

	/**
	 * Your task should go here. Make a new function for each of the
	 * tasks. You will also want to create windows for the restored 
	 * images and display them. The function GaussianBlur shows you
	 * how convolution can be implemented.
	 */

	Sobel(coin1);

	cv::waitKey();

	return 0;
}

void Sobel(cv::Mat &input)
{
	// Create kernels
	cv::Mat kernelX = (cv::Mat_<double>(3,3) << -1, 0, 1, -2, 0, 2, -1,  0,  1);
	cv::Mat kernelY = (cv::Mat_<double>(3,3) <<  1, 2, 1,  0, 0, 0, -1, -2, -1);

	// we need to create a padded version of the input
	// or there will be border effects
	int kernelRadiusX = ( kernelX.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernelX.size[1] - 1 ) / 2;

	// create the temporary kernel
	cv::Mat tempX, tempY;
	tempX.create(input.size(), CV_64FC1);
	tempY.create(input.size(), CV_64FC1);

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );
	// now we can do the convoltion
	for ( int i = 0; i < input.rows; i++ )
	{	
		for( int j = 0; j < input.cols; j++ )
		{
			double sumX = 0.0;
			double sumY = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
			{
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
				{
					// find the correct indices we are using
					int imagex = i + 1 + m;
					int imagey = j + 1 + n;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalvalX = kernelX.at<double>( kernelx, kernely );
					double kernalvalY = kernelY.at<double>( kernelx, kernely );
				
					// do the multiplication
					sumX += imageval * kernalvalX;		
					sumY += imageval * kernalvalY;				
				}
			}
			// set the output value as the sum of the convolution
			tempX.at<double>(i,j) = (double)sumX;
			tempY.at<double>(i,j) = (double)sumY;

		}
	}

	// Magnitude
	cv::Mat magnitude;
	magnitude.create(input.size(), CV_64FC1);
	magnitude = tempX.mul(tempX) + tempY.mul(tempY);
	cv::sqrt(magnitude, magnitude);

	// Angle
	cv::Mat angle;
	angle.create(input.size(), CV_64FC1);
	for ( int i = 0; i < input.rows; i++ )
	{	
		for( int j = 0; j < input.cols; j++ )
		{
			angle.at<double>(i,j) = atan2(tempY.at<double>(i,j),tempX.at<double>(i,j));
		}
	}

	//angle = atan2(tempY/tempX);

	cv::normalize(tempX, tempX, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::normalize(tempY, tempY, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::normalize(angle, angle, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	/* Display the images */
	cv::imshow("X", tempX);
	cv::imshow("Y", tempY);
	cv::imshow("Magnitude", magnitude);
	cv::imshow("angle", angle);
}
