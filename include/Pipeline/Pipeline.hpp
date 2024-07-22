// Daniele Fabris

#ifndef PIPELINE_HPP
#define PIEPELINE_HPP

#include "FieldSegmentor/FieldSegmentor.hpp"
#include "BallsDetector/BallsDetector.hpp"
#include "Metrics/Metrics.hpp"

#include "Utils/Utils.hpp"

#include <opencv2/opencv.hpp>

/**
 * Structure to save the program relevant output.
 */
struct PipelineExecOutput
{
    std::vector<Utils::BallBoundingBox> boundingBoxes_ff;       // Bounding boxes of first frame.
    std::vector<Utils::BallBoundingBox> boundingBoxes_fl;       // Bounding boxes of last frame.
    cv::Mat binMask_ff;                                         // Bin mask of first frame.
    cv::Mat binMask_fl;                                         // Bin mask of last frame.
    cv::Mat colorMask_ff;                                       // Color mask of first frame.
    cv::Mat colorMask_fl;                                       // Color mask of last frame.
    cv::Mat output_first_ff;                                    // First output of first frame.
    cv::Mat output_first_fl;                                    // First output of last frame.
};

struct PipelineEvalOutput
{
    float mIoU_ff;
    float mIoU_fl;
    // float mAP_ff;
    // float mAP_fl;
};

class Pipeline
{
    private:
    std::string clip_;                  // VideoCapture file path
    cv::Mat frame_first_;
    cv::Mat frame_last_;

    std::string gt_bb_ff_path_;         // Ground truth bounding boxes first frame file path
    std::string gt_bb_fl_path_;         // Ground truth bounding boxes last frame file path
    
    std::string gt_sm_ff_path_;         // Ground truth segmentation mask first frame file path
    std::string gt_sm_fl_path_;         // Ground truth segmentation mask last frame file path

    FieldSegmentor fieldSegmentor_;
    BallsDetector ballsDetector_;
    Metrics metrics_;

    std::string outputFolder_;          // Output folder path
    
    public:

    /**
     * Pipeline constructor.
     * @param clip VideoCapture.
     * @param frame_first First frame image.
     * @param frame_last Last frame image.
     * @param gt_bb_ff_path Path to the ground truth of the bounding boxes of first frame.
     * @param gt_bb_fl_path Path to the ground truth of the bounding boxes of last frame.
     * @param gt_sm_ff_path Path to the ground truth masks of segmentation of first frame.
     * @param gt_sm_lf_path Path to the ground truth masks of segmentation of last frame.
     */
    Pipeline(std::string clip_path, const cv::Mat& frame_first, const cv::Mat& frame_last, 
            std::string gt_bb_ff_path, std::string gt_bb_fl_path, 
            std::string gt_sm_ff_path, std::string gt_sm_fl_path,
            std::string outputFolder);

    /**
     * Executes the pipeline on the given input.
     * @return The PipelineExecOutput structure containing the results.
     */
    PipelineExecOutput exec();

    /**
     * Evaluates the output of the pipeline execution.
     * @return The PipelineEvalOutput structure containing the evaluation.
     */
    PipelineEvalOutput eval(PipelineExecOutput output);

};


#endif  // PIPELINE_HPP