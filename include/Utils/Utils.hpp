// Daniele Fabris

#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <map>
#include <tuple>
#include <cmath>
#include <limits>

namespace Utils
{
    /**
     * Struct to store information about the corner.
     */
    struct Corner
    {
        cv::Point point;                                    // Point where the corner is situated.
        std::pair<cv::Vec4i, cv::Vec4i> generatingLines;    // Pair of lines generating the corner.
    };

    /**
     * Struct to store information about detected box's coordinates.
     */
    struct DetectedBox
    {
        int x;          // x-coordinate of top-left corner
        int y;          // y-coordinate of top-left corner
        int width;      // width of the bounding box
        int height;     // height of the bounding box
    };

    struct BallBoundingBox : DetectedBox
    {
        int ID;             // ball category ID
    };

    /* Comparator of cv::Vec3b couple. */
    struct Vec3bCompare {
        bool operator()(const cv::Vec3b& a, const cv::Vec3b& b) const;
    };

    /**
     * Check if two colors are similar within a given threshold.
     * @param color1 First color.
     * @param color2 Second color.
     * @param threshold Channels difference threshold.
     * @return True if the pair of colors is similar within the threshold, otherwise false.
     */
    bool similarColors(const cv::Vec3b& color1, const cv::Vec3b& color2, int threshold);

    /**
     * Check if two color are similar in the HSV color space.
     * @param color1 First color in HSV.
     * @param color2 SecondColor in HSV.
     * @param hueThreshold Threshold of Hue channel.
     * @param satThreshold Threshold of Saturation channel.
     * @param valThreshold Threshold of Value channel.
     * @return True if the pair of colors is similar within the given thresholds.
     */
    bool similarColorsHSV(cv::Vec3b& color1, cv::Vec3b& color2, int hueThreshold, int satThreshold, int valThreshold);

    /**
     * Check if the colors are different (redundant).
     * @param color First color.
     * @param tableColor Color of the table.
     * @param threshold Threshold.
     * @return True if the colors are different.
     */
    bool isDifferentColor(const cv::Vec3b& color, const cv::Vec3b& tableColor, int threshold);

    /**
     * Convert the image to HSV color space, compute the histogram and find the prominent color.
     * @param img The input image.
     * @return The estimated prominent color in HSV space.
     */
    cv::Vec3b estimateTableColorHSV(const cv::Mat& img);

    /**
     * Coompute the BGR prominent color using kmeans clustering.
     * @param img The input image.
     * @param k Number of clusters.
     * @return The estimated prominent color in BGR space.
     */
    cv::Vec3b estimateTableColorBGR(const cv::Mat& img, int k);
    
    /**
     * Filter the region of the pool table in an image base on HSV color space.
     * @param img The input image.
     * @param colorHSV The estimated prominent HSV color of the table.
     * @return A binary mask where white pixels indicate the segmented table.
     */
    cv::Mat getFieldMaskHSV(const cv::Mat& img, const cv::Vec3b& colorHSV);
    
    /**
     * Filter the region of the pool table in an image base on BGR color space.
     * @param img The input image.
     * @param colorHSV The estimated prominent BGR color of the table.
     * @param threshold Threshold parameter.
     * @return A binary mask where white pixels indicate the segmented table.
     */
    cv::Mat getFieldMaskBGR(const cv::Mat& img, const cv::Vec3b& colorBGR, int threshold);

    /**
     * Function to compute gradients.
     * @param dx Partial derivate of image in X direction.
     * @param dy Partial derivate of image in Y direction.
     * @param edges Edges given by Canny.
     * @param gradients Vector of points representing computed gradients.
     */
    void calculateGradients(const cv::Mat& dx, const cv::Mat& dy, std::vector<cv::Point>& edges, std::vector<cv::Point>& gradients);

    /**
     * Compute angle of a line segment.
     * @param line Segment line.
     * @return Angle of the line.
     */  
    double computeAngle(cv::Vec4i& line);

    /**
     * Compute distance between two points.
     * @param a First point.
     * @param b Second point.
     * @return Distance between points.
     */
    double pointsDistance(cv::Point2f a, cv::Point2f b);

    /**
     * Compute the squared distance between two points.
     * @param p1 First point.
     * @param p2 Second point.
     * @return The squared distance.
     */
    double squaredDistance(const cv::Point& p1, const cv::Point& p2);

    /**
     * Compute middle point of a line.
     * @param line Segment line.
     * @return Middle point of the line.
     */  
    cv::Point computeMiddlePoint(cv::Vec4i& line);

    /**
     * Check if two lines are the same line.
     * @param line1 First line.
     * @param line2 Second line.
     * @return True if the two lines are the same.
     */
    bool areLinesSame(const cv::Vec4i& line1, const cv::Vec4i& line2);

    /**
     * Read the bounding boxes from a file.
     * @param path The path to the file.
     * @return The vector of balls bounding boxes.
     */
    std::vector<BallBoundingBox> readBoundingBoxesFromFile(std::string path);

    /**
     * Write the bounding boxes to a file.
     * @param bboxes The bounding boxes to write.
     * @param path The path of the file.
     */
    void writeBoundingBoxesToFile(const std::vector<BallBoundingBox>& bboxes, std::string path);

};  // Utils

#endif  // UTILS_HPP