// Daniele Fabris

#include "utils/Utils.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

namespace Utils
{

    bool Vec3bCompare::operator()(const cv::Vec3b& a, const cv::Vec3b& b) const
    {
        return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
    }

    bool similarColors(const cv::Vec3b& color1, const cv::Vec3b& color2, int threshold)
    {
        return std::abs(color1[0] - color2[0]) <= threshold &&
            std::abs(color1[1] - color2[1]) <= threshold &&
            std::abs(color1[2] - color2[2]) <= threshold;
    }

    bool similarColorsHSV(cv::Vec3b& color1, cv::Vec3b& color2, int hueThreshold, int satThreshold, int valThreshold)
    {
        int hueDiff = abs(color1[0] - color2[0]);
        int satDiff = abs(color1[1] - color2[1]);
        int valDiff = abs(color1[2] - color2[2]);

        hueDiff = std::min(hueDiff, 180 - hueDiff);

        return (hueDiff <= hueThreshold && satDiff <= satThreshold && valDiff <= valThreshold);
    }

    bool isDifferentColor(const cv::Vec3b& color, const cv::Vec3b& tableColor, int threshold = 50) {
        return (abs(color[0] - tableColor[0]) > threshold ||
                abs(color[1] - tableColor[1]) > threshold ||
                abs(color[2] - tableColor[2]) > threshold);
    }

    cv::Vec3b estimateTableColorHSV(const cv::Mat& img)
    {
        cv::Mat imgHSV;
        cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

        int hBins = 30, sBins = 32;
        int histSize[] = {hBins, sBins};
        float hRanges[] = {0, 180};
        float sRanges[] = {0, 256};
        const float* ranges[] = {hRanges, sRanges};
        cv::Mat hist;

        int channels[] = {0, 1};
        cv::calcHist(&imgHSV, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
        double maxVal = 0;
        cv::Point maxLoc;
        cv::minMaxLoc(hist, 0, &maxVal, 0, &maxLoc);

        cv::Vec3b prominentColor;
        prominentColor[0] = maxLoc.y * (180 / hBins);
        prominentColor[1] = maxLoc.x * (256 / sBins);
        prominentColor[2] = 255; // for convenience
        std::cout << "Prominent color HSV: (" << (int)prominentColor[0] << ", " << (int)prominentColor[1] << ", " << (int)prominentColor[2] << ")" << std::endl;
        
        return prominentColor;
    }

    cv::Vec3b estimateTableColorBGR(const cv::Mat& img, int k)
    {
        cv::Mat samples(img.rows * img.cols, 3, CV_32F);
        for (int y = 0; y < img.rows; ++y)
        {
            for (int x = 0; x < img.cols; ++x)
            {
                for (int z = 0; z < 3; ++z)
                {
                    samples.at<float>(y + x * img.rows, z) = img.at<cv::Vec3b>(y, x)[z];
                }
            }
        }           

        cv::Mat labels, centers;
        int attempts = 5;
        cv::kmeans(samples, k, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10000, 0.0001), attempts, cv::KMEANS_PP_CENTERS, centers);

        std::vector<int> counts(k, 0);
        for (int i = 0; i < labels.rows; ++i)
        {
            counts[labels.at<int>(i, 0)]++;
        }

        int maxCounter = 0;
        int dominantCluster = 0;
        for (int i = 0; i < k; ++i)
        {
            if (counts[i] > maxCounter)
            {
                maxCounter = counts[i];
                dominantCluster = i;
            }
        }

        cv::Vec3b prominentColor;
        prominentColor[0] = (uchar)centers.at<float>(dominantCluster, 0);
        prominentColor[1] = (uchar)centers.at<float>(dominantCluster, 1);
        prominentColor[2] = (uchar)centers.at<float>(dominantCluster, 2);

        return prominentColor;
    }

    cv::Mat getFieldMaskHSV(const cv::Mat& img, const cv::Vec3b& colorHSV)
    {
        cv::Mat imgHSV, mask;
        cv::cvtColor(img, imgHSV, cv::COLOR_BGR2HSV);

        int hue = colorHSV[0];
        int saturation = colorHSV[1];
        int hueRange = 15;
        int saturationRange = 35;

        cv::Scalar lowerBound(std::max(0, hue - hueRange), std::max(0, saturation - saturationRange), 0);
        cv::Scalar upperBound(std::min(180, hue + hueRange), std::min(255, saturation + saturationRange), 255);

        cv::inRange(imgHSV, lowerBound, upperBound, mask);

        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
        
        return mask;
    }

    cv::Mat getFieldMaskBGR(const cv::Mat& img, const cv::Vec3b& colorBGR, int threshold)
    {
        cv::Mat mask;
        cv::inRange(img, colorBGR - cv::Vec3b(threshold, threshold, threshold), colorBGR + cv::Vec3b(threshold, threshold, threshold), mask);
        
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

        return mask;
    }

    void calculateGradients(const cv::Mat& dx, const cv::Mat& dy, std::vector<cv::Point>& edges, std::vector<cv::Point>& gradients)
    {
        for (const cv::Point& edge : edges)
        {
            int x = edge.x;
            int y = edge.y;
            float gradientX = dx.at<float>(y, x);
            float gradientY = dy.at<float>(y, x);
            gradients.push_back(cv::Point(gradientX, gradientY));
        }
    }

    double computeAngle(cv::Vec4i& line)
    {
        double dy = static_cast<double>(line[3] - line[1]);
        double dx = static_cast<double>(line[2] - line[0]);
        return std::atan2(dy, dx) * 180.0 / CV_PI;
    }

    double pointsDistance(cv::Point2f a, cv::Point2f b)
    {
        return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
    }

    double squaredDistance(const cv::Point& p1, const cv::Point& p2)
    {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        return dx * dx + dy * dy;
    }

    cv::Point computeMiddlePoint(cv::Vec4i& line)
    {
        return cv::Point((line[0] + line[2]) / 2.0f, (line[1] + line[3]) / 2.0f);
    }

    bool areLinesSame(const cv::Vec4i& line1, const cv::Vec4i& line2)
    {
        return (line1 == line2) || (line1[0] == line2[2] && line1[1] == line2[3] && line1[2] == line2[0] && line1[3] == line2[1]);
    }

    std::vector<BallBoundingBox> readBoundingBoxesFromFile(std::string path)
    {
        std::vector<BallBoundingBox> boundingBoxes;
        std::ifstream file(path);
        std::string line;

        if (!file.is_open())
        {
            std::cerr << "Error: Could not open the file: " << path << std::endl;
            return boundingBoxes;
        }

        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            BallBoundingBox bbox;
            if (!(iss >> bbox.x >> bbox.y >> bbox.width >> bbox.height >> bbox.ID))
            {
                std::cerr << "Error: Culd not read line: " << line << std::endl;
                continue;
            }
            boundingBoxes.push_back(bbox);
        }
        file.close();
        return boundingBoxes;
    }

    void writeBoundingBoxesToFile(const std::vector<BallBoundingBox>& bboxes, std::string path)
    {
        std::ofstream output(path);
        if (!output.is_open())
        {
            std::cerr << "Error: Could not open the output file." << std::endl;
            return;
        }

        for (const auto& bbox : bboxes)
        {
            if (bbox.ID != -1)
            {
                output << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height << " " << bbox.ID << std::endl;
            }
        }
        output.close();
    }

};  // Utils