// Daniele Fabris

#include <Metrics/Metrics.hpp>
#include "utils/Utils.hpp"

#include <opencv2/core/core.hpp>
#include <vector>

float Metrics::computeGeometricIoU(const Utils::BallBoundingBox &bb1, const Utils::BallBoundingBox &bb2)
{
    int overlapX = std::max(0, std::min(bb1.x + bb1.width, bb2.x + bb2.width) - std::max(bb1.x, bb2.x));
    int overlapY = std::max(0, std::min(bb1.y + bb1.height, bb2.y + bb2.height) - std::max(bb1.y, bb2.y));

    int overlapArea = overlapX * overlapY;
    int unionArea = bb1.width * bb1.height + bb2.width * bb2.height - overlapArea;

    if (unionArea == 0) return 0;
    return (float)overlapArea / unionArea;
}

float Metrics::computeClassIoU(const cv::Mat& predicted, const cv::Mat& gt, int label)
{
    int intersect = 0, un = 0;

    for (int i = 0; i < predicted.rows; ++i)
    {
        for (int j = 0; j < predicted.cols; ++j)
        {
            if (predicted.at<uchar>(i, j) == label && gt.at<uchar>(i, j) == label) {
                intersect++;
                un++;
            }
            else if (predicted.at<uchar>(i, j) == label && gt.at<uchar>(i, j) != label ||
                     predicted.at<uchar>(i, j) != label &&  gt.at<uchar>(i, j) == label)
            {
                un++;
            }
        }
    }
    if (un == 0) un++;

    return (float)intersect / un;
}

float Metrics::computeMIoU(const cv::Mat& predicted, const cv::Mat& gt)
{
    if (!(predicted.size() == gt.size() && predicted.type() == gt.type()))
    {
        std::cerr << "Predicted and ground truth can't be compared." << std::endl;
        return 0;
    }
    /** There are 5 classes:
     * 0 - Background
     * 1 - White "cue ball"
     * 2 - Black "8-ball"
     * 3 - Ball with solid color
     * 4 - Ball with stripes
     * 5 - Playing field (table).
     * */
    float IoU0 = computeClassIoU(predicted, gt, 0);
    float IoU1 = computeClassIoU(predicted, gt, 1);
    float IoU2 = computeClassIoU(predicted, gt, 2);
    float IoU3 = computeClassIoU(predicted, gt, 3);
    float IoU4 = computeClassIoU(predicted, gt, 4);
    float IoU5 = computeClassIoU(predicted, gt, 5);

    return (IoU0 + IoU1 + IoU2 + IoU3 + IoU4 + IoU5) / 6;

}