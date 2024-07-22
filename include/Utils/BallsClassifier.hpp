// Daniele Fabris

#ifndef BALLS_CLASSIFIER_HPP
#define BALLS_CLASSIFIER_HPP

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

namespace BallsClassifier
{

    enum BallType
    {
        CUE,
        EIGHT,
        SOLID,
        STRIPED
    };

    /**
     * Verify if inside the ball ROI there is tha white ball.
     * @param ballROI The ROI in input image corresponding to the ball's area.
     * @return True if the ball is white.
     */
    bool isWhiteBall(const cv::Mat& ballROI);

    /**
     * Verify if inside the ball ROI there is tha black ball.
     * @param ballROI The ROI in input image corresponding to the ball's area.
     * @return True if the ball is black.
     */
    bool isBlackBall(const cv::Mat& ballROI);

    /**
     * Find which type the ball is, checking its mean brightness and counting the proportion
     * of white pixel w.r.t. the total number of pixels.
     * @param ballROI The ROI in input image corresponding to the ball's area.
     * @param brightnessThs Brightness threshold.
     * @param whiteRatio Ratio of white pixels to identify striped ball.
     * @return The ball type.
     */
    BallType classifyBall(const cv::Mat& ballROI, double brightnessThs, double whiteRatio);

};  // BallsClassifier

#endif  // BALLS_CLASSIFIER_HPP