#include <iostream>
#include <opencv.hpp>
#include <highgui/highgui.hpp>
#include <imgproc/imgproc.hpp>
#include <math.h>

#define M_PI 3.141592654

using namespace cv;

/**
 * Function to add gaussian blur to an image
 *
 * Requires: An input image to be blured
 *           An output image that will contain the blurred input
 *           A kernel size
 */
void Sobel(
           Mat &input, int T);

void Hough(Mat &magnitude, Mat &gradient);

void Threshold(Mat &magnitude, int threshold);

int main ( int argc, char ** argv )
{
	/* Load up all of the images of car numberplates */
 	Mat coin1 = imread("coins1.png", CV_LOAD_IMAGE_GRAYSCALE);
    
	/**
	 * Your task should go here. Make a new function for each of the
	 * tasks. You will also want to create windows for the restored
	 * images and display them. The function GaussianBlur shows you
	 * how convolution can be implemented.
	 */
    Sobel(coin1, atoi(argv[1]));
    
    waitKey();
    
    return 0;
}

void Hough(Mat &magnitude, Mat &gradient)
{
    int width = magnitude.size().width;
    int height = magnitude.size().height;
    
    const int size[] = {width, height, 1};
    int i, j, x, y;
    
    double houghSpace[width][height][1];
    
    int radius = 10;
    
    for (i = 0; i < magnitude.rows; i++ )
    {
        for(j = 0; j < magnitude.cols; j++ )
        {
            if(magnitude.at<uchar>( i, j ) == 255)
            {
                for (int theta = 0; theta < 360; theta++) {
                    x = i - radius * cos(gradient.at<uchar>(i, j));
                    y = i - radius * sin(gradient.at<uchar>(i, j));
                    if ((x < width) && (x > 0) && (y < height) && (y > 0)) {
                        houghSpace[i][j][0] += 1;
                    }
                }
            }
        }
    }
}

void Threshold(Mat &magnitude, int threshold)
{
    for ( int i = 0; i < magnitude.rows; i++ )
    {
        for( int j = 0; j < magnitude.cols; j++ )
        {
            if (magnitude.at<uchar>( i, j ) > threshold)
                magnitude.at<uchar>( i, j ) = 255;
            else
                magnitude.at<uchar>( i, j ) = 0;
        }
    }
    
    imshow("Threshold", magnitude);
}

void Sobel(Mat &input, int T)
{
	// Create kernels
    Mat kernelX = (Mat_<double>(3,3) << -1, 0, 1, -2, 0, 2, -1,  0,  1);
    Mat kernelY = (Mat_<double>(3,3) <<  1, 2, 1,  0, 0, 0, -1, -2, -1);
    
	// we need to create a padded version of the input
	// or there will be border effects
    int kernelRadiusX = ( kernelX.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernelX.size[1] - 1 ) / 2;
    
	// create the temporary kernel
    Mat tempX, tempY;
    tempX.create(input.size(), CV_64FC1);
    tempY.create(input.size(), CV_64FC1);
    
    Mat paddedInput;
    copyMakeBorder( input, paddedInput,
                   kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                   BORDER_REPLICATE );
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
    Mat magnitude;
    magnitude.create(input.size(), CV_64FC1);
    magnitude = tempX.mul(tempX) + tempY.mul(tempY);
    sqrt(magnitude, magnitude);
    
	// Angle
    Mat angle;
    angle.create(input.size(), CV_64FC1);
    for ( int i = 0; i < input.rows; i++ )
    {
        for( int j = 0; j < input.cols; j++ )
        {
            angle.at<double>(i,j) = atan2(tempY.at<double>(i,j),tempX.at<double>(i,j));
        }
    }
    
    normalize(tempX, tempX, 0, 255, NORM_MINMAX, CV_8UC1);
    normalize(tempY, tempY, 0, 255, NORM_MINMAX, CV_8UC1);
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8UC1);
    normalize(angle, angle, 0, 255, NORM_MINMAX, CV_8UC1);
    
	/* Display the images
    imshow("X", tempX);
    imshow("Y", tempY);
     */
    imshow("Magnitude", magnitude);
    imshow("angle", angle);
    
    Threshold (magnitude, T);
    
    /* Apply the Hough transform using the magnitude and gradient images found */
    Hough(magnitude, angle);
}
