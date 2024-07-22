// Daniele Fabris

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

#include <opencv2/flann.hpp>

#include "BallsDetector/BallsDetector.hpp"
#include "Utils/Utils.hpp"
#include "Utils/BallsClassifier.hpp"

std::vector<Utils::DetectedBox> BallsDetector::checkBoundingBoxesPosition(const std::vector<Utils::DetectedBox>& boxes, const std::vector<Utils::Corner>& corners)
{
    std::vector<Utils::DetectedBox> filteredBoxes;
    std::vector<cv::Point> polygon;
    for (const auto& corner : corners)
    {
        polygon.push_back(cv::Point(corner.point));
    }
    /**
     * Filter the bounding boxes in order to don't pick false positives on the edges, 
     * like pockets of playing field.
     *  */
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        Utils::DetectedBox box = boxes[i];
        cv::Point boxCenter(box.x + box.height / 2, box.y + box.width / 2);
        if (cv::pointPolygonTest(polygon, boxCenter, true) > 20.0)
        {
            filteredBoxes.push_back(box);
        }
    }
    return filteredBoxes;
}

std::vector<Utils::DetectedBox> BallsDetector::createBoundingBoxes(const std::vector<cv::Vec3f>& circles)
{
    std::vector<Utils::DetectedBox> boxes;
    /**
     * For each circle create the DetectedBox (x, y, wifth, height).
     */
    for (size_t i = 0; i < circles.size(); ++i)
    {
        cv::Vec3f circle = circles[i];
        cv::Point center = cv::Point(cvRound(circle[0]), cvRound(circle[1]));
        int radius = cvRound(circle[2]);

        Utils::DetectedBox box;
        box.x = center.x - radius;
        box.y = center.y - radius;
        box.width = 2 * radius;
        box.height = 2 * radius;

        boxes.push_back(box);
    }
    return boxes;
}

cv::Mat BallsDetector::preprocessImage(const cv::Mat& src)
{
    cv::Mat hsv_image, lab_image, combined_image, blurred_image, thresh;

    /* Convert the image in HSV and Lab color spaces. */
    cv::cvtColor(src, hsv_image, cv::COLOR_BGR2HSV);
    cv::cvtColor(src, lab_image, cv::COLOR_BGR2Lab);

    /* Extract V channel from the image in HSV color space. */
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv_image, hsv_channels);
    cv::Mat& v_channel = hsv_channels[2];

    /* Extract L channel from the image in Lab color space. */
    std::vector<cv::Mat> lab_channels;
    cv::split(lab_image, lab_channels);
    cv::Mat& l_channel = lab_channels[0];  // Canale L

    /* Combine channel V of HSV and L of Lab. */
    cv::addWeighted(v_channel, 0.5, l_channel, 0.5, 0, combined_image);

    /* Apply Gaussian Blur to reduce noise. */
    cv::GaussianBlur(combined_image, blurred_image, cv::Size(5, 5), 0);

    /* Apply adaptive thresholding on the combined channel. */
    cv::adaptiveThreshold(blurred_image, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 10);

    return thresh;
}

std::vector<cv::Vec3f> BallsDetector::filterCircles(const cv::Mat& img, const std::vector<cv::Vec3f>& circles, double distanceThreshold, double radiusThreshold)
{
    std::vector<cv::Vec3f> uniqueCircles;
    /* Removing duplicates circles, that are close circles with similar radii. */
    for (size_t i = 0; i < circles.size(); ++i)
    {
        bool isUnique = true;
        for (size_t j = 0; j < uniqueCircles.size(); ++j)
        {
            double dist = Utils::pointsDistance(cv::Point2f(circles[i][0], circles[i][1]), cv::Point2f(uniqueCircles[j][0], uniqueCircles[j][1]));
            if (dist < distanceThreshold && std::abs(circles[i][2] - uniqueCircles[j][2]) < radiusThreshold)
            {
                isUnique = false;
                break;
            }
        }
        if (isUnique) {
            uniqueCircles.push_back(circles[i]);
        }
    }
    return uniqueCircles;
}

float BallsDetector::euclideanDistance(cv::Point2f p1, cv::Point2f p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

bool BallsDetector::isCircular(const std::vector<cv::Point2f>& points, cv::Vec3f& circle) {
    /* The size of the set of points is too low, we can't check if it's circular. */
    if (points.size() < 5) {
        return false;
    }

    cv::Point2f center;
    float radius;
    cv::minEnclosingCircle(points, center, radius);

    /* Compute the mean squared error. */
    float mse = 0.0;
    for (const cv::Point2f& point : points) {
        mse += pow(euclideanDistance(point, center) - radius, 2);
    }
    mse /= points.size();

    /* Check if the error is acceptable to consider the set a circle. */
    const float maxMSE = 20.0;
    if (mse <= maxMSE) {
        circle = cv::Vec3f(center.x, center.y, radius);
        return true;
    } else {
        return false;
    }
}

void BallsDetector::findAndSaveCircles(const cv::Mat& binaryImage, std::vector<cv::Vec3f>& circles, float minArea, float maxArea) {
    cv::Mat tempImage = binaryImage.clone();

    /* Find the contours in the binary image.  */
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(tempImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const std::vector<cv::Point>& contour : contours) {
        /* Compute the area of contour (cluster). */
        float area = cv::contourArea(contour);

        /* Checl if the area is between the thresholds. */
        if (area >= minArea && area <= maxArea) {
            std::vector<cv::Point2f> points;
            for (const cv::Point& point : contour) {
                points.push_back(cv::Point2f(point.x, point.y));
            }

            /* Check the circularity of the points. */
            cv::Vec3f circle;
            if (isCircular(points, circle)) {
                circles.push_back(circle);
            }
        }
    }
}

std::vector<cv::Vec3f> BallsDetector::findCircles(const cv::Mat& img, cv::Mat& mask, const std::vector<Utils::Corner>& corners)
{
    /* Invert field mask. */
    cv::Mat invertedMask;
    cv::bitwise_not(mask, invertedMask);
    /* Take in consideration only the part of the image delimited by the corners of the field. */
    cv::Mat filter = cv::Mat::zeros(mask.size(), CV_8UC1);
    std::vector<cv::Point> points;
    for (auto& corner : corners)
    {
        points.push_back(corner.point);
    }
    cv::fillConvexPoly(filter, {points}, cv::Scalar(255));
    cv::bitwise_and(invertedMask, filter, invertedMask);

    cv::Mat image = cv::Mat(img.rows, img.cols, CV_8UC3, img.type());
    image.setTo(cv::Scalar(0,0,0));
    img.copyTo(image, filter);
    
    /* Preprocess imamge. */
    cv::Mat preprocessed = preprocessImage(image);

    cv::Mat combined;
    cv::bitwise_or(invertedMask, preprocessed, combined);

    /* Double method to find circles. */
    std::vector<cv::Vec3f> circles;
    findAndSaveCircles(combined, circles, 100, 400);

    std::vector<cv::Vec3f> houghCircles;
    cv::HoughCircles(combined, houghCircles, cv::HOUGH_GRADIENT, 2, 10, 100, 30, 8, 20);
    for (cv::Vec3f c : houghCircles)
    {
        circles.push_back(c);
    }

    return circles;
}

std::vector<Utils::BallBoundingBox> BallsDetector::detectBalls(const cv::Mat& img, cv::Mat& mask, const std::vector<Utils::Corner>& corners)
{
    cv::Mat image = img.clone();
    std::vector<cv::Vec3f> detectedCircles, filteredCircles;
    /* Find circles in the image. */
    detectedCircles = findCircles(img, mask, corners);
    double distanceThs = 15.0, radiusThs = 5.0;
    filteredCircles = filterCircles(img, detectedCircles, distanceThs, radiusThs);
    /* Create the bounding boxes for the detected circles. */
    std::vector<Utils::DetectedBox> initialBoxes;
    initialBoxes = createBoundingBoxes(filteredCircles);
    /* Keep only the bounding boxes inside the playing field. */
    std::vector<Utils::DetectedBox> boundingBoxes;
    boundingBoxes = checkBoundingBoxesPosition(initialBoxes, corners);
    /* Create the final bounding boxes, with even ID of the ball specified. */
    std::vector<Utils::BallBoundingBox> finalBoxes;
    for (size_t i = 0; i < boundingBoxes.size(); i++) {
        cv::Rect r;
        r.x = boundingBoxes[i].x;
        r.y = boundingBoxes[i].y;
        r.width = boundingBoxes[i].width;
        r.height = boundingBoxes[i].height;

        Utils::BallBoundingBox finalBox;
        finalBox.x = boundingBoxes[i].x;
        finalBox.y = boundingBoxes[i].y;
        finalBox.width = boundingBoxes[i].width;
        finalBox.height = boundingBoxes[i].height;

        cv::Mat ballROI = img(r);
        BallsClassifier::BallType type;
        double brightnessThs = 0.7, whitePercentage = 0.1;
        type = BallsClassifier::classifyBall(ballROI, brightnessThs, whitePercentage);
        switch (type)
        {
            case BallsClassifier::CUE:
                finalBox.ID = 1;
                break;
            case BallsClassifier::EIGHT:
                finalBox.ID = 2;
                break;
            case BallsClassifier::SOLID:
                finalBox.ID = 3;
                break;
            case BallsClassifier::STRIPED:
                finalBox.ID = 4;
                break;
        }
        finalBoxes.push_back(finalBox);
    }

    return finalBoxes;
}

void BallsDetector::drawBoundingBoxes(cv::Mat&img, std::vector<Utils::BallBoundingBox>& boxes)
{
    for (const auto& box : boxes)
    {
        cv::Scalar color;
        int type = box.ID;
        switch (type) 
        {
            case 1:
                color = cv::Scalar(255, 255, 255);
                break;
            case 2:
                color = cv::Scalar(0, 0, 0);
                break;
            case 3:
                color = cv::Scalar(255, 0, 0);
                break;
            case 4:
                color = cv::Scalar(0, 0, 255);
                break;
        }
        cv::Rect roi(box.x, box.y, box.width, box.height);

        cv::Mat overlay;
        img.copyTo(overlay);

        cv::rectangle(overlay, roi, color, cv::FILLED);
        /* Opacity of rect. */
        double alpha = 0.5;
        cv::addWeighted(overlay, alpha, img, 1 - alpha, 0, img);

        /* Draw the border of rectangle. */
        cv::rectangle(img, roi, color, 1);
    }
}

void BallsDetector::getFinalBinColorMask(const cv::Mat& img, const std::vector<Utils::BallBoundingBox>& bboxes, cv::Mat& binMask, cv::Mat& colorMask)
{
    for (auto& box : bboxes)
    {
        int id = box.ID;
        cv::Vec3b color;
        switch (id)
        {
            case 1:
                color = cv::Vec3b(255, 255, 255);
                break;
            case 2:
                color = cv::Vec3b(0, 0, 0);
                break;
            case 3:
                color = cv::Vec3b(255, 0, 0);
                break;
            case 4:
                color = cv::Vec3b(0, 0, 255);
                break;
        }
        float radius = box.width / 2;
        float x = box.x + box.width / 2;
        float y = box.y + box.height / 2;
        cv::Point2f center(x, y);

        cv::circle(binMask, center, radius, cv::Scalar(id), cv::FILLED);
        cv::circle(colorMask, center, radius, color, cv::FILLED);
    }
}