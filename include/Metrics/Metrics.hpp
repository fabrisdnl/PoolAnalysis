// Daniele Fabris

#ifndef METRICS_HPP
#define METRICS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "Utils/Utils.hpp"

/**
 * Metrics class for mIoU computations.
 */
class Metrics {

   private:
    /**
     * Computes the geometric Intersection over Union (IoU) between two bounding boxes.
     * @param bb1 First bounding box.
     * @param bb2 Second bounding box.
     * @return The computed IoU.
     */
    float computeGeometricIoU(const Utils::BallBoundingBox& bb1, const Utils::BallBoundingBox& bb2);

    /**
     * Computes the IoU for a single class.
     * @param predicted Predicted labels.
     * @param gt Ground truth labels.
     * @param label Class label for which IoU must be computed.
     * @return The computed IoU value for the class.
     */
    float computeClassIoU(const cv::Mat& predicted, const cv::Mat& gt, int label);

    public:
    /**
     * Computes the mean Intersection over Union (mIoU) for segmentation task.
     * @param predicted predicted.
     * @param gt Ground truth labels.
     * @return The computed mIoU value.
     */
    float computeMIoU(const cv::Mat& predicted, const cv::Mat& gt);

};

#endif  // METRICS_HPP