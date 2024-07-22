// Daniele Fabris

#include "Utils/BallsClassifier.hpp"
#include <iostream>

namespace BallsClassifier
{

    bool isWhiteBall(const cv::Mat& ballROI)
    {
        int white_pixel_count = 0;
        int total_pixels = ballROI.rows * ballROI.cols;
        
        for (int i = 0; i < ballROI.rows; ++i)
        {
            for (int j = 0; j < ballROI.cols; ++j)
            {
                cv::Vec3b pixel = ballROI.at<cv::Vec3b>(i, j);
                if (pixel[0] > 180 && pixel[1] > 180 && pixel[2] > 150)
                {
                    white_pixel_count++;
                }
            }
        }

        double white_pixel_ratio = static_cast<double>(white_pixel_count) / total_pixels;
        return white_pixel_ratio > 0.4;
    }

    bool isBlackBall(const cv::Mat& ballROI)
    {
        int black_pixel_count = 0;
        int total_pixels = ballROI.rows * ballROI.cols;
        
        for (int i = 0; i < ballROI.rows; ++i)
        {
            for (int j = 0; j < ballROI.cols; ++j)
            {
                cv::Vec3b pixel = ballROI.at<cv::Vec3b>(i, j);
                if (pixel[0] < 50 && pixel[1] < 50 && pixel[2] < 50)
                {
                    black_pixel_count++;
                }
            }
        }

        double black_pixel_ratio = static_cast<double>(black_pixel_count) / total_pixels;
        return black_pixel_ratio > 0.25;
    }

    BallType classifyBall(const cv::Mat& ballROI, double brightnessThs, double whiteRatio)
    {
        /* Check if it's the white ball. */
        if (isWhiteBall(ballROI))
        {
            return CUE;
        }
        /* CHeck if it's the black ball. */
        if (isBlackBall(ballROI))
        {
            return EIGHT;
        }

        /* Convert ROI to grayscale. */
        cv::Mat gray;
        cv::cvtColor(ballROI, gray, cv::COLOR_BGR2GRAY);

        /* Apply threshold to create a binary image. */
        cv::Mat binary;
        cv::threshold(gray, binary, brightnessThs * 255, 255, cv::THRESH_BINARY);

        /* Count white pixels. */
        int whitePixels = cv::countNonZero(binary);
        int totalPixels = binary.rows * binary.cols;
        double whitePixelRatio = static_cast<double>(whitePixels) / totalPixels;
        double cueRatio = 0.3;

        /* Classify based on the white pixel ratio. */
        if (whitePixelRatio > cueRatio)
        {
            return CUE;
        }
        else if (whitePixelRatio > whiteRatio)
        {
            return STRIPED;
        }
        else 
        {
            return SOLID;
        }
    }

};  // BallsClassifier
