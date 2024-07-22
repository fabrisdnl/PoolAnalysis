// Daniele Fabris

#ifndef FIELD_SEGMENTOR_HPP
#define FIELD_SEGMENTOR_HPP

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Utils/Utils.hpp"

class FieldSegmentor
{
    private:

    /**
     * Find the intersection of two lines.
     * @param a First line.
     * @param b Second line.
     * @return Intersection point cv::Point2f.
     */
    cv::Point2f computeIntersection(cv::Vec4i a, cv::Vec4i b);
    
    /**
     * Extend a line to the image borders.
     * @param mask The mask of the table's field.
     * @param line The given line to be extended.
     * @return A cv::Vec4i line extended to the borders.
     */
    cv::Vec4i extendLine(const cv::Mat& mask, const cv::Vec4i& line);

    /**
     * Generates the next combinations of elements.
     * @param first First iterator.
     * @param k Second iterator.
     * @param last Last iterator.
     * @return True if generates a combination.
     */
    bool nextCombination(std::vector<int>::iterator first, std::vector<int>::iterator k, std::vector<int>::iterator last);

    /**
     * Counts the number of white and black pixels in the polygon.
     * @param mask The mask segmenting the table's playing field.
     * @param corners A vector of cv::Point consisting in the corners of the polygon.
     * @return The pair of values (whitePixels, blackPixels).
     */
    std::pair<int, int> countPixelsInPolygon(const cv::Mat& mask, const std::vector<Utils::Corner>& corners);

    /**
     * Compute the convex hull of corners.
     * @param corners Vector of corners.
     * @return Conves hull of these points.
     */
    std::vector<Utils::Corner> convexHull(const std::vector<Utils::Corner>& corners);

    /**
     * Order points in clockwise order.
     * @param points Points to be ordered.
     * @return Ordered vector of points (clockwise).
     */
    std::vector<Utils::Corner> orderPointsClockwise(const std::vector<Utils::Corner>& points);

    /**
     * Find the best 4 corners by maximizing the area of white pixels within the polygon.
     * @param possibleCorners Vector of possible corners to be examinated.
     * @param mask The mask segmenting the table's playing field.
     * @return The optimal corners.
     */
    std::vector<Utils::Corner> selectOptimalCorners(const std::vector<Utils::Corner>& possibleCorners, const cv::Mat& mask);

    /**
     * Find the intersections of lines from different groups.
     * @param groupedLines The groups of lines.
     * @param imgSize The image size.
     * @return A vector of possible corners of the playing field.
     */
    std::vector<Utils::Corner> findPossibleCorners(const std::vector<std::vector<cv::Vec4i>> groupedLines, const cv::Size& imgSize);

    /**
     * Group lines with similar angles and close proximity.
     * @param lines Vector of lines.
     * @param angleThreshold Threshold angle.
     * @param distanceThreshold Threshold distance.
     * @return Groups of lines close to each other and with similar angles.
     */
    std::vector<std::vector<cv::Vec4i>> groupLines(const std::vector<cv::Vec4i>& lines, float angleThreshold, float distanceThreshold);

    /**
     * Find the corners of the table's field, and ignore pixels outside.
     * @param mask The input mask segmenting approximately the table's field.
     * @return The corners of the playing field.
     */
    std::vector<Utils::Corner> findCorners(const cv::Mat& mask);

    public:

    /**
     * Find the corners of playing field to segment the table's field.
     * @param img The input image.
     * @return The mask used and a vector of cv::Point representing corners of the field.
     */
    std::pair<cv::Mat, std::vector<Utils::Corner>> getPlayingField(const cv::Mat& img);

    /**
     * Draw the contours of table's field in the input image.
     * @param img The input image.
     * @param corners The corners of the playing field.
     */
    void drawContours(const cv::Mat& img, const std::vector<Utils::Corner>& corners);

    /**
     * Get initial bin mask, identifying background and playing field.
     * @param img The input image.
     * @param fieldMask The binary mask of the field.
     * @param corners The corners of the playing field.
     * @param binMask Initial bin segmentation mask.
     * @param colorMask Initial color segmentation mask.
     */
    void getBinColorMaskField(const cv::Mat& img, const cv::Mat& fieldMask, const std::vector<Utils::Corner>& corners, cv::Mat& binMask, cv::Mat& colorMask);

};

#endif  // FIELD_SEGMENTOR_HPP