// Daniele Fabris

#ifndef BALLS_DETECTOR_HPP
#define BALLS_DETECTOR_HPP

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Utils/Utils.hpp"

/**
 * BallsDetector class responsible for detecting the balls on the pool table.
 */
class BallsDetector
{
    private:

    /**
     * Check if the bounding boxes are contained in the playing field.
     * @param boxes The vector of bounding boxes.
     * @param corners The corners of the playing field.
     * The bounding boxes contained in the playing field.
     */
    std::vector<Utils::DetectedBox> checkBoundingBoxesPosition(const std::vector<Utils::DetectedBox>& boxes, const std::vector<Utils::Corner>& corners);

    /**
     * Create a bounding box for each circle found.
     * @param circles The vector of circles found.
     * @return The vector of bounding boxes.
     */
    std::vector<Utils::DetectedBox> createBoundingBoxes(const std::vector<cv::Vec3f>& circles);

    /**
     * Preprocess image on HSV and Lab color spaces.
     * @param src The image to preprocess.
     * @return The image preprocessed.
     */
    cv::Mat preprocessImage(const cv::Mat& src);

    /**
     * Compute the euclidean distance between two points.
     * @param p1 The first point.
     * @param p2 The second point.
     * @return The distance.
     */
    float euclideanDistance(cv::Point2f p1, cv::Point2f p2);

    /**
     * Check the circularity of a set of points.
     * @param points The points to be checked.
     * @param circle A circle where is going to be saved the circle representing the circular set of points,
     * if the circle is sufficiently circular.
     * @return True if the set of points is circular.
     */
    bool isCircular(const std::vector<cv::Point2f>& points, cv::Vec3f& circle);

    /**
     * Find and save the circles of the binary image.
     * @param binaryImage The binary image to be analyzed.
     * @param circles Vector where the circles are going to be saved.
     * @param minArea The minimum area of the cluster of points.
     * @param maxArea The maximum area of the cluster of points.
     */
    void findAndSaveCircles(const cv::Mat& binaryImage, std::vector<cv::Vec3f>& circles, float minArea, float maxArea);

    /**
     * Filter duplicate circles.
     * @param img The input image.
     * @param circles Vector of circles to be filtered.
     * @param distanceThreshold Threshold of distance between centers.
     * @param radiusThreshold Threshold of difference between radius of circles.
     * @return The vector of unique circles.
     */
    std::vector<cv::Vec3f> filterCircles(const cv::Mat& img, const std::vector<cv::Vec3f>& circles, double distanceThreshold, double radiusThreshold);

    /**
     * Find all the circles in the image.
     * @param img The input image.
     * @param mask Mask of the playing field.
     * @param corners The vector of corners of the playing field.
     * @return A vector of cv::Vec3f, one for each circle found in the image.
     */
    std::vector<cv::Vec3f> findCircles(const cv::Mat& img, cv::Mat& mask, const std::vector<Utils::Corner>& corners);

    public:

    /**
     * BallsDetector's class main method to detect balls on the pool table.
     * @param img The input image.
     * @param mask The mask of the playing field.
     * @param corners The corners of the playing field
     * @return A vector of BallBoundingBox containing each of the ball found on the pool table.
     */
    std::vector<Utils::BallBoundingBox> detectBalls(const cv::Mat& img, cv::Mat& mask, const std::vector<Utils::Corner>& corners);

    /**
     * Draw the bounding boxes around each ball detected.
     * @param img The image where the boxes are going to be drawn.
     * @param boxes The bounding boxes of the detected balls.
     */
    void drawBoundingBoxes(cv::Mat& img, std::vector<Utils::BallBoundingBox>& boxes);

    /**
     * Get final bin mask, identifying balls.
     * @param img The input image.
     * @param bboxes The bounding boxes of the balls.
     * @param binMask Initial bin segmentation mask.
     * @param colorMask Initial color segmentation mask.
     */
    void getFinalBinColorMask(const cv::Mat& img, const std::vector<Utils::BallBoundingBox>& bboxes, cv::Mat& binMask, cv::Mat& colorMask);

};

#endif  // BALLS_DETECTOR_HPP