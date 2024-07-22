// Daniele Fabris

#include "Pipeline/Pipeline.hpp"
#include "FieldSegmentor/FieldSegmentor.hpp"
#include "BallsDetector/BallsDetector.hpp"
#include "BallsTracker/BallsTracker.hpp"

#include "Utils/Utils.hpp"

#include <iostream>
#include <map>
#include <tuple>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

PipelineEvalOutput Pipeline::eval(PipelineExecOutput output)
{
    /* Compute metrics. */
    PipelineEvalOutput evaluation;

    cv::Mat gt_sm_ff = cv::imread(gt_sm_ff_path_, cv::IMREAD_GRAYSCALE);
    cv::Mat gt_sm_fl = cv::imread(gt_sm_fl_path_, cv::IMREAD_GRAYSCALE);
    
    if (gt_sm_ff.empty() || gt_sm_fl.empty())
    {
        std::cerr << "Error: Could not open ground truth segmentation masks." << std::endl;
        evaluation.mIoU_ff = 0.0;
        evaluation.mIoU_fl = 0.0;
    }
    else
    {
        evaluation.mIoU_ff = metrics_.computeMIoU(output.binMask_ff, gt_sm_ff);
        evaluation.mIoU_fl = metrics_.computeMIoU(output.binMask_fl, gt_sm_fl);
    }

    /* I'm not using any kind of machine learning, so computing mAP does not have any sense. */
    // std::vector<Utils::BallBoundingBox> gt_bb_ff = Utils::readBoundingBoxesFromFile(gt_bb_ff_path_);
    // std::vector<Utils::BallBoundingBox> gt_bb_fl = Utils::readBoundingBoxesFromFile(gt_bb_fl_path_);
    
    return evaluation;
}

PipelineExecOutput Pipeline::exec()
{
    PipelineExecOutput output;

    std::vector<Utils::BallBoundingBox> boundingBoxes_ff, boundingBoxes_fl;
    cv::Mat binMask_ff, binMask_fl, colorMask_ff, colorMask_fl;
    cv::Mat output_first_ff, output_second_ff, output_first_fl, output_second_fl;

    std::vector<Utils::BallBoundingBox> detectedBoxes_ff, detectedBoxes_fl;

    /* Step 1: Segment the field and detect balls for first frame. */
    auto [fieldMask_ff, fieldCorners_ff] = fieldSegmentor_.getPlayingField(frame_first_);

    output_first_ff = frame_first_.clone();
    output_second_ff = frame_first_.clone();

    fieldSegmentor_.drawContours(output_first_ff, fieldCorners_ff);
    
    detectedBoxes_ff = ballsDetector_.detectBalls(frame_first_, fieldMask_ff, fieldCorners_ff);    

    ballsDetector_.drawBoundingBoxes(output_first_ff, detectedBoxes_ff);
    
    fieldSegmentor_.getBinColorMaskField(frame_first_, fieldMask_ff, fieldCorners_ff, binMask_ff, colorMask_ff);
    ballsDetector_.getFinalBinColorMask(frame_first_, detectedBoxes_ff, binMask_ff, colorMask_ff);

    fieldSegmentor_.drawContours(colorMask_ff, fieldCorners_ff);

    output_second_ff = colorMask_ff;
    
    /* Step 2: Segment the field and detect balls for last frame. */
    auto [fieldMask_fl, fieldCorners_fl] = fieldSegmentor_.getPlayingField(frame_last_);    

    output_first_fl = frame_last_.clone();
    output_second_fl = frame_last_.clone();

    fieldSegmentor_.drawContours(output_first_fl, fieldCorners_fl);
    
    detectedBoxes_fl = ballsDetector_.detectBalls(frame_last_, fieldMask_fl, fieldCorners_fl);

    ballsDetector_.drawBoundingBoxes(output_first_fl, detectedBoxes_fl);

    fieldSegmentor_.getBinColorMaskField(frame_last_, fieldMask_fl, fieldCorners_fl, binMask_fl, colorMask_fl);
    ballsDetector_.getFinalBinColorMask(frame_last_, detectedBoxes_fl, binMask_fl, colorMask_fl);

    fieldSegmentor_.drawContours(colorMask_fl, fieldCorners_fl);

    output_second_fl = colorMask_fl;

    output.boundingBoxes_ff = detectedBoxes_ff;
    output.boundingBoxes_fl = detectedBoxes_fl;
    output.binMask_ff = binMask_ff;
    output.binMask_fl = binMask_fl;
    output.colorMask_ff = colorMask_ff;
    output.colorMask_fl = colorMask_fl;
    output.output_first_ff = output_first_ff;
    output.output_first_fl = output_first_fl;

    /* Step 3: Balls trajectories tracking. */
    BallsTracker ballsTracker_(detectedBoxes_ff, fieldCorners_ff, outputFolder_);
    if (!ballsTracker_.trajectoryTracking(clip_))
    {
        std::cerr << "Problem with the videocapture." << std::endl;
    }

    return output;
}

Pipeline::Pipeline(std::string clip, const cv::Mat& frame_first, const cv::Mat& frame_last, 
    std::string gt_bb_ff_path, std::string gt_bb_fl_path, 
    std::string gt_sm_ff_path, std::string gt_sm_fl_path,
    std::string outputFolder)
    :   clip_(clip), frame_first_(frame_first), frame_last_(frame_last),
        gt_bb_ff_path_(gt_bb_ff_path), gt_bb_fl_path_(gt_bb_fl_path),
        gt_sm_ff_path_(gt_sm_ff_path), gt_sm_fl_path_(gt_sm_fl_path), 
        outputFolder_(outputFolder),
        fieldSegmentor_(), ballsDetector_(), metrics_() {}
