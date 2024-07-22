// Daniele Fabris

#include <iostream>
#include <unistd.h>
#include <string>

#include <opencv2/opencv.hpp>

#include "Pipeline/Pipeline.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <Path to clip folder>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string clip_folder_path = argv[1];
    int lastSlashPos = clip_folder_path.rfind('/');

    std::string clip_path = clip_folder_path + clip_folder_path.substr(lastSlashPos) + ".mp4";
    std::string frame_first_path = clip_folder_path + "/frames/frame_first.png";
    std::string frame_last_path = clip_folder_path + "/frames/frame_last.png";
    std::string gt_bb_ff_path = clip_folder_path + "/bounding_boxes/frame_first_bbox.txt";
    std::string gt_bb_fl_path = clip_folder_path + "/bounding_boxes/frame_last_bbox.txt";
    std::string gt_sm_ff_path = clip_folder_path + "/masks/frame_first.png";;
    std::string gt_sm_fl_path = clip_folder_path + "/masks/frame_last.png";

    std::cout << clip_path << std::endl;

    cv::Mat frame_first, frame_last;
    frame_first = cv::imread(frame_first_path, cv::IMREAD_COLOR);
    frame_last = cv::imread(frame_last_path, cv::IMREAD_COLOR);
    if (frame_first.empty() || frame_last.empty())
    {
        std::cerr << "Error: Could not open the first and last frames." << std::endl;
        return -1;
    }

    std::string outputFolder = "../output";

    Pipeline pipeline(clip_path, frame_first, frame_last, gt_bb_ff_path, gt_bb_fl_path,  gt_sm_ff_path, gt_sm_fl_path, outputFolder);
    
    std::cout << "EXECUTION STARTED" << std::endl;

    PipelineExecOutput execOutput = pipeline.exec();

    std::cout << "EVALUATION STARTED" << std::endl;

    PipelineEvalOutput evalOutput = pipeline.eval(execOutput);

    std::cout << "First Frame mIoU: " << evalOutput.mIoU_ff << std::endl;
    std::cout << "Last Frame mIoU: " << evalOutput.mIoU_fl << std::endl;
    
    /* Save in output the results. */

    std::cout << "SAVING FILES" << std::endl;

    std::string bbFolder = outputFolder + "/bounding_boxes";
    std::string masksFolder = outputFolder + "/masks";

    Utils::writeBoundingBoxesToFile(execOutput.boundingBoxes_ff, bbFolder + "/frame_first_bboxes.txt");
    Utils::writeBoundingBoxesToFile(execOutput.boundingBoxes_fl, bbFolder + "/frame_last_bboxes.txt");

    cv::imwrite(masksFolder + "/frame_first_bin_mask.png", execOutput.binMask_ff);
    cv::imwrite(masksFolder + "/frame_first_color_mask.png", execOutput.colorMask_ff);
    cv::imwrite(masksFolder + "/frame_last_bin_mask.png", execOutput.binMask_fl);
    cv::imwrite(masksFolder + "/frame_last_color_mask.png", execOutput.colorMask_fl);

    cv::imwrite(masksFolder + "/output_first_ff.png", execOutput.output_first_ff);
    cv::imwrite(masksFolder + "/output_first_fl.png", execOutput.output_first_fl);
    
    std::cout << "COMPLETED!" << std::endl;

    cv::waitKey();

    return 0;
}
