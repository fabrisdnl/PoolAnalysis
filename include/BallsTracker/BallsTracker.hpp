// Daniele Fabris

#ifndef BALLS_TRACKER_HPP
#define BALLS_TRACKER_HPP

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

#include "Utils/Utils.hpp"

/**
 * Structure in order to save relevant information useful to ball drawing.
 */
struct Ball2D
{
    int id;                                 // ID of the ball.
    cv::Rect bbox;                          // Bounding box.
    cv::Ptr<cv::Tracker> tracker;           // Tracker
    std::vector<cv::Point2f> trajectory;    // Trajectory
};

class BallsTracker
{
    private:
    std::vector<Utils::BallBoundingBox> bboxes_;
    std::vector<cv::Point2f> corners_;
    std::string outputFolder_;

    /**
     * Draw a trajectory.
     * @param image The cv::Mat where the trajectory must be drawn.
     * @param trajectory The trajectory.
     * @param color The color of trajectory.
     */
    void drawTrajectory(cv::Mat& image, const std::vector<cv::Point2f>& trajectory, const cv::Scalar& color);

    /**
     * Update the 2D map with the current positions of balls and their trajectories.
     * @param minimap The cv::Mat map.
     * @param balls The vector of Ball2D structures.
     * @param pockets The holes of the billiard table.
     */
    void update2DMap(cv::Mat& minimap, const std::vector<Ball2D>& balls, const std::vector<cv::Point2f>& pockets);

    public:

    /**
     * Represent the current state of the game in a 2D top-view visualization map, 
     * to be updated at each new frame with the current ball positions and the 
     * trajectory of each ball that is moving.
     * @param clip The input videocapture.
     * @return False if a problem occurs.
     */
    bool trajectoryTracking(std::string clip_path);

    /* Constructor. */
    BallsTracker(const std::vector<Utils::BallBoundingBox>& boxes, const std::vector<Utils::Corner>& corners, std::string outputFolder);
};

#endif