// Daniele Fabris

#include "FieldSegmentor/FieldSegmentor.hpp"
#include "Utils/Utils.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>

cv::Point2f FieldSegmentor::computeIntersection(cv::Vec4i a, cv::Vec4i b)
{
    cv::Point2f point;
    float x1 = a[0], y1 = a[1], x2 = a[2], y2=a[3];
    float x3 = b[0], y3 = b[1], x4 = b[2], y4=b[3];

    float d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (d != 0)
    {
        point.x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / d;
        point.y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / d;
    }
    else point = cv::Point2f(-1, -1);
    
    return point;
}

cv::Vec4i FieldSegmentor::extendLine(const cv::Mat& mask, const cv::Vec4i& line)
{
    float x1 = line[0], y1 = line[1];
    float x2 = line[2], y2 = line[3];
    
    float dx = x2 - x1;
    float dy = y2 - y1;

    cv::Point p1, p2;

    if (dx == 0) // Vertical line
    {
        p1 = cv::Point(x1, 0);
        p2 = cv::Point(x1, mask.size().height - 1);
    } 
    if (dy == 0) // Horizontal line
    {
        p1 = cv::Point(0, y1);
        p2 = cv::Point(mask.size().width - 1, y1);
    }
    else
    {
        float slope = dy / dx;
        float intercept = y1 - slope * x1;

        cv::Point points[4];
        points[0] = cv::Point(0, intercept);                                                            // Left border
        points[1] = cv::Point(mask.size().width - 1, slope * (mask.size().width - 1) + intercept);      // Right border
        points[2] = cv::Point(- intercept / slope, 0);                                                  // Top border
        points[3] = cv::Point((mask.size().height - 1 - intercept) / slope, mask.size().height - 1);    // Bottom border
        
        std::vector<cv::Point> validPoints;
        for (cv::Point p : points)
        {
            if (p.x >= 0 && p.y >= 0 && p.x < mask.size().width && p.y < mask.size().height)
            {
                validPoints.push_back(p);
            }
        }
        if (validPoints.size() >= 2)
        {
            p1 = validPoints[0];
            p2 = validPoints[1];
        }
    }
    return cv::Vec4i(p1.x, p1.y, p2.x, p2.y);
}

bool FieldSegmentor::nextCombination(std::vector<int>::iterator first, std::vector<int>::iterator k, std::vector<int>::iterator last)
{
    if ((first == last) || (first == k) || (last == k)) return false;
    std::vector<int>::iterator i1 = first;
    std::vector<int>::iterator i2 = last;
    ++i1;
    if (last == i1) return false;
    i1 = last;
    --i1;
    i1 = k;
    --i2;
    while (first != i1)
    {
        if (*--i1 < *i2)
        {
            std::vector<int>::iterator j = k;
            while (!(*i1 < *j)) ++j;
            std::iter_swap(i1, j);
            ++i1;
            ++j;
            i2 = k;
            std::rotate(i1, j, last);
            while (last != j)
            {
                ++j;
                ++i2;
            }
            std::rotate(k, i2, last);
            return true;
        }
    }
    std::rotate(first, k, last);
    return false;
}

std::pair<int, int> FieldSegmentor::countPixelsInPolygon(const cv::Mat& mask, const std::vector<Utils::Corner>& corners)
{
    cv::Mat polyMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    std::vector<cv::Point> polygon;
    for (const auto& corner : corners)
    {
        polygon.push_back(corner.point);
    }
    std::vector<std::vector<cv::Point>> polygons = {polygon};
    cv::fillPoly(polyMask, polygons, cv::Scalar(255));
    /* Count the white and black pixels inside the polygon delimited by corners. */
    int whitePixels = 0, blackPixels = 0;
    for (int y = 0; y < mask.rows; ++y)
    {
        for (int x = 0; x < mask.cols; ++x)
        {
            if (polyMask.at<uchar>(y, x) == 255)
            {
                if (mask.at<uchar>(y, x) == 255) whitePixels++;
                else blackPixels++;
            }
        }
    }

    return {whitePixels, blackPixels};    
}

std::vector<Utils::Corner> FieldSegmentor::convexHull(const std::vector<Utils::Corner>& corners)
{
    /* Extract points from corners. */
    std::vector<cv::Point> points;
    for (const auto& corner : corners)
    {
        points.push_back(corner.point);
    }
    /* Compute convex hull. */
    std::vector<int> hullIndices;
    cv::convexHull(points, hullIndices);
    /* Map the hull indices back to the original corners. */
    std::vector<Utils::Corner> hullCorners;
    for (int index : hullIndices)
    {
        hullCorners.push_back(corners[index]);
    }
    return hullCorners;
}

std::vector<Utils::Corner> FieldSegmentor::orderPointsClockwise(const std::vector<Utils::Corner>& points)
{
    /* Compute the centroid. */
    cv::Point centroid(0,0);
    for (const auto& p : points)
    {
        centroid.x += p.point.x;
        centroid.y += p.point.y;
    }
    centroid.x /= points.size();
    centroid.y /= points.size();
    /* Order the points in clockwise order, considering the centroid. */
    std::vector<Utils::Corner> orderedPoints = points;
    std::sort(orderedPoints.begin(), orderedPoints.end(), [&centroid](const Utils::Corner& a, const Utils::Corner& b)
    {
        float angleA = std::atan2(a.point.y - centroid.y, a.point.x - centroid.x);
        float angleB = std::atan2(b.point.y - centroid.y, b.point.x - centroid.x); 
        return angleA < angleB;
    });
    return orderedPoints;
}

std::vector<Utils::Corner> FieldSegmentor::selectOptimalCorners(const std::vector<Utils::Corner>& possibleCorners, const cv::Mat& mask)
{
    std::vector<Utils::Corner> optimalCorners;
    if (possibleCorners.size() > 4)
    {
        int maxWhitePixels = 0;
        int minBlackPixels = std::numeric_limits<int>::max();

        std::vector<int> indices(possibleCorners.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::vector<int> combination(4);
        do
        {
            for (int i = 0; i < 4; ++i)
            {
                combination[i] = indices[i];
            }
            std::vector<Utils::Corner> polygon = {possibleCorners[combination[0]], possibleCorners[combination[1]], possibleCorners[combination[2]], possibleCorners[combination[3]]};
            bool validPolygon = true;
            for (size_t j = 0; j < polygon.size(); ++j)
            {
                size_t next_j = (j + 1) % polygon.size();
                const auto& currentCorner = polygon[j];
                const auto& nextCorner = polygon[next_j];
                bool foundConnectingLine = Utils::areLinesSame(currentCorner.generatingLines.first, nextCorner.generatingLines.first) ||
                                           Utils::areLinesSame(currentCorner.generatingLines.first, nextCorner.generatingLines.second) ||
                                           Utils::areLinesSame(currentCorner.generatingLines.second, nextCorner.generatingLines.first) ||
                                           Utils::areLinesSame(currentCorner.generatingLines.second, nextCorner.generatingLines.second);
                if (!foundConnectingLine)
                {
                    validPolygon = false;
                    break;
                }
            }
            if (validPolygon)
            {
                auto [whitePixels, blackPixels] = countPixelsInPolygon(mask, polygon);

                if (whitePixels > maxWhitePixels || (whitePixels == maxWhitePixels && blackPixels < minBlackPixels))
                {
                    maxWhitePixels = whitePixels;
                    minBlackPixels = blackPixels;
                    optimalCorners = polygon;
                }
            } 
        } while (nextCombination(indices.begin(), indices.begin() + 4, indices.end()));
    }
    else optimalCorners = possibleCorners;
    return optimalCorners;
}

std::vector<Utils::Corner> FieldSegmentor::findPossibleCorners(const std::vector<std::vector<cv::Vec4i>> groupedLines, const cv::Size& imgSize)
{
    std::vector<Utils::Corner> possibleCorners;
    for (size_t i = 0; i < groupedLines.size(); ++i)
    {
        for (size_t j = i + 1; j < groupedLines.size(); ++j)
        {
            for (const auto& line1 : groupedLines[i])
            {
                for (const auto& line2 : groupedLines[j])
                {
                    cv::Point pt = computeIntersection(line1, line2);
                    if (pt.x >= 0 && pt.y >= 0 && pt.x < imgSize.width && pt.y < imgSize.height)
                    {
                        bool isDuplicate = false;
                        for (const auto& existingCorner : possibleCorners)
                        {
                            if (cv::norm(pt - existingCorner.point) < 1.0)
                            {
                                isDuplicate = true;
                                break;
                            }
                        }
                        if (!isDuplicate) possibleCorners.push_back({pt, {line1, line2}});
                    }
                }
            }
        }
    }
    return possibleCorners;
}

std::vector<std::vector<cv::Vec4i>> FieldSegmentor::groupLines(const std::vector<cv::Vec4i>& lines, float angleThreshold, float distanceThreshold)
{
    /* Extract angles and middle points of lines for clustering. */
    cv::Mat data(lines.size(), 2, CV_32F);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        float angle = std::atan2(lines[i][3] - lines[i][1], lines[i][2] - lines[i][0]);
        cv::Point middlePoint((lines[i][0] + lines[i][2]) / 2.0f, (lines[i][1] + lines[i][3]) / 2.0f);
        data.at<float>(i, 0) = angle;
        data.at<float>(i, 1) = middlePoint.x;
        data.at<float>(i, 2) = middlePoint.y;
    }
    /* Apply k-means clustering to group lines in 4 clusters. */
    int k = 4;
    cv::Mat labels, centers;
    cv::kmeans(data, k, labels, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);
    /* Order the clusters by number of lines (decreasing). */
    std::vector<std::vector<cv::Vec4i>> groupedLines(k);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        int clusterIndex = labels.at<int>(i);
        groupedLines[clusterIndex].push_back(lines[i]);
    }
    std::sort(groupedLines.begin(), groupedLines.end(), [](const std::vector<cv::Vec4i>& a, const std::vector<cv::Vec4i>& b)
    {
        return a.size() > b.size();
    });
    size_t numberToTake = std::min(groupedLines.size(), static_cast<size_t>(4));
    std::vector<std::vector<cv::Vec4i>> firstFourGroups(groupedLines.begin(), groupedLines.begin() + numberToTake);
    return firstFourGroups;
}

std::vector<Utils::Corner> FieldSegmentor::findCorners(const cv::Mat& mask)
{
    /* Apply Canny Edge Detector to the mask. */
    cv::Mat edges;
    cv::Canny(mask, edges, 50, 150, 3);
    /* Find lines using Hough Line Transform. */
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 100, 30, 5);
    std::cout << "Computed Hough Line Transform: " << lines.size() << std::endl;
    /* Extend the lines found to the image borders. */
    std::vector<cv::Vec4i> extendedLines;
    for (size_t i = 0; i < lines.size(); ++i)
    {
        cv::Vec4i extendedLine = extendLine(mask, lines[i]);
        extendedLines.push_back(extendedLine);
    }
    std::cout << "Extended Lines: " << extendedLines.size() << std::endl;
    /* Group lines with similar angles and close proximity. */
    std::vector<std::vector<cv::Vec4i>> groupedLines;
    groupedLines = groupLines(lines, CV_PI / 36, 30);
    std::cout << "Grouped Lines: " << groupedLines.size() << std::endl;
    /* Find possible corners analyzing the groups of lines. */
    std::vector<Utils::Corner> possibleCorners;
    possibleCorners = findPossibleCorners(groupedLines, mask.size());
    std::cout << "Possible Corners: " << possibleCorners.size() << std::endl;
    /* Use convex hull to filter corners. */
    std::vector<Utils::Corner> hull;
    hull = convexHull(possibleCorners);
    /* Sort corners in clockwise order. */
    std::vector<Utils::Corner> orderedCorners;
    orderedCorners = orderPointsClockwise(possibleCorners);
    std::cout << "Ordered corners: " << orderedCorners.size() << std::endl;
    /* Find the best 4 corners by maximizing the area of white pixels within the polygon formed by them. */
    std::vector<Utils::Corner> optimalCorners;
    optimalCorners = selectOptimalCorners(orderedCorners, mask);
    std::cout << "Optimal Corners: " << optimalCorners.size() << std::endl;
    return optimalCorners;
}

std::pair<cv::Mat, std::vector<Utils::Corner>> FieldSegmentor::getPlayingField(const cv::Mat& img)
{
    /* To estimate color crop the image to select central part. */
    int border = 200;
    int croppedWidth = img.cols - 2 * border;
    int croppedHeight = img.rows - 2 * border;
    cv::Rect roi(border, border, croppedWidth, croppedHeight);
    cv::Mat croppedImg = img(roi);

    cv::Vec3b prominentColorHSV, prominentColorBGR;
    prominentColorHSV = Utils::estimateTableColorHSV(croppedImg);
    prominentColorBGR = Utils::estimateTableColorBGR(croppedImg, 5);
    cv::Mat maskHSV, maskBGR, mask;
    maskHSV = Utils::getFieldMaskHSV(img, prominentColorHSV);
    maskBGR = Utils::getFieldMaskBGR(img, prominentColorBGR, 30);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(maskHSV, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> hulls(contours.size());
    for (size_t i = 0; i < contours.size(); ++i)
    {
        cv::convexHull(contours[i], hulls[i]);
    }
    double maxArea = 0;
    std::vector<cv::Point> largestHull;
    for (size_t i = 0; i < hulls.size(); ++i)
    {
        double area = cv::contourArea(hulls[i]);
        if (area > maxArea)
        {
            maxArea = area;
            largestHull = hulls[i];
        }
    }

    cv::Mat temp = cv::Mat::zeros(img.size(), CV_8UC1);

    std::vector<std::vector<cv::Point>> largestHullVec = {largestHull};
    cv::fillPoly(temp, largestHullVec, cv::Scalar(255));

    cv::Mat result;
    cv::bitwise_and(maskHSV, temp, result);
    cv::bitwise_or(result, maskBGR, result);

    /* Find the corners of the playing field. */
    std::vector<Utils::Corner> corners;
    corners = findCorners(result);
    return {result, corners};
}

void FieldSegmentor::drawContours(const cv::Mat& img, const std::vector<Utils::Corner>& corners)
{
    for (size_t i = 0; i < corners.size(); ++i)
    {
        size_t next_i = (i + 1) % corners.size();
        cv::Vec4i line = Utils::areLinesSame(corners[i].generatingLines.first, corners[next_i].generatingLines.first) ?
            corners[i].generatingLines.first : corners[i].generatingLines.second;
        
        cv::line(img, corners[i].point, corners[next_i].point, cv::Scalar(0, 255, 255), 2);
    }
}

void FieldSegmentor::getBinColorMaskField(const cv::Mat& img, const cv::Mat& fieldMask, const std::vector<Utils::Corner>& corners, cv::Mat& binMask, cv::Mat& colorMask)
{
    binMask = cv::Mat::zeros(img.size(), CV_8UC1);
    colorMask = img.clone();

    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

    cv::Vec3b field(0, 255, 0);

    std::vector<cv::Point> points;
    for (size_t i = 0; i < corners.size(); ++i)
    {
        points.push_back(corners[i].point);
    }
    std::vector<std::vector<cv::Point>> polygons = {points};

    cv::fillPoly(mask, polygons, cv::Scalar(255));

    /* Color mask based on the polygon delimited by corners of the playing field. */
    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            uchar maskValue = mask.at<uchar>(y, x);
            if (maskValue == 0)
            {
                continue;
            }
                        colorMask.at<cv::Vec3b>(y, x) = field;
        }
    }
    /* Bin mask based on playing field mask. */
    for (int y = 0; y < img.rows; ++y)
    {
        for (int x = 0; x < img.cols; ++x)
        {
            uchar fieldMaskValue = fieldMask.at<uchar>(y, x);
            if (fieldMaskValue == 0)
            {
                continue;
            }
            
            binMask.at<uchar>(y, x) = 5;
        }
    }
}