// Daniele Fabris

#include "BallsTracker/BallsTracker.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include <iostream>

void BallsTracker::drawTrajectory(cv::Mat& image, const std::vector<cv::Point2f>& trajectory, const cv::Scalar& color)
{
    for (size_t i = 1; i < trajectory.size(); ++i)
    {
        cv::line(image, trajectory[i - 1], trajectory[i], color, 1);
    }
}

void BallsTracker::update2DMap(cv::Mat& minimap, const std::vector<Ball2D>& balls, const std::vector<cv::Point2f>& pockets)
{
    minimap.setTo(cv::Scalar(255, 255, 255));

    /* Draws lines of the tables. */
    cv::line(minimap, pockets[0], pockets[1], cv::Scalar(0, 0, 0), 2);
    cv::line(minimap, pockets[1], pockets[2], cv::Scalar(0, 0, 0), 2);
    cv::line(minimap, pockets[2], pockets[3], cv::Scalar(0, 0, 0), 2);
    cv::line(minimap, pockets[3], pockets[4], cv::Scalar(0, 0, 0), 2);
    cv::line(minimap, pockets[4], pockets[5], cv::Scalar(0, 0, 0), 2);
    cv::line(minimap, pockets[5], pockets[0], cv::Scalar(0, 0, 0), 2);

    /* Draw internal lines of the tables. */
    // int offset = 10;
    // std::vector<cv::Point2f> innerPoints = {
    //     {pockets[0].x + offset, pockets[0].y + offset},
    //     {pockets[1].x, pockets[1].y + offset},
    //     {pockets[2].x - offset, pockets[2].y + offset},
    //     {pockets[3].x - offset, pockets[3].y - offset},
    //     {pockets[4].x - offset, pockets[4].y - offset},
    //     {pockets[5].x + offset, pockets[5].y - offset}
    // };

    // cv::line(minimap, innerPoints[0], innerPoints[1], cv::Scalar(0, 0, 0), 2);
    // cv::line(minimap, innerPoints[1], innerPoints[2], cv::Scalar(0, 0, 0), 2);
    // cv::line(minimap, innerPoints[2], innerPoints[3], cv::Scalar(0, 0, 0), 2);
    // cv::line(minimap, innerPoints[3], innerPoints[4], cv::Scalar(0, 0, 0), 2);
    // cv::line(minimap, innerPoints[4], innerPoints[5], cv::Scalar(0, 0, 0), 2);
    // cv::line(minimap, innerPoints[5], innerPoints[0], cv::Scalar(0, 0, 0), 2);

    /* Draw the holes. */
    for (const auto &pocket : pockets)
    {
        cv::circle(minimap, pocket, 20, cv::Scalar(0, 0, 0), -1);
    }

    /* Draw each ball detected and its trajectory. */    
    for (const auto &ball : balls)
    {
        if (!ball.trajectory.empty())
        {
            cv::Scalar color;
            switch (ball.id)
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
            drawTrajectory(minimap, ball.trajectory, cv::Scalar(255, 0, 255));
            cv::circle(minimap, ball.trajectory.back(), 10, color, -1);
            cv::circle(minimap, ball.trajectory.back(), 10, cv::Scalar(0, 0, 0), 1);
        }
    }
}

bool BallsTracker::trajectoryTracking(std::string clip_path)
{
    cv::VideoCapture cap(clip_path);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video clip" << std::endl;
        return false;
    }
    /* If not even one ball has been detected, don't do anything. */
    if (bboxes_.size() < 1)
    {
        std::cout << "No bounding boxes." << std::endl;
        return false;
    }

    /* Compute the homography. */
    std::vector<cv::Point2f> dstPoints = {cv::Point2f(50, 50), cv::Point2f(750, 50), cv::Point2f(750, 350), cv::Point2f(50, 350)};
    cv::Mat homography = cv::findHomography(corners_, dstPoints);
    /* Variables to track the trajectories. */
    std::vector<std::vector<cv::Point2f>> trajectories(bboxes_.size());
    /* Top 2D visualization map. */
    cv::Mat minimap(400, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    /* Holes of the playing field in the minimap. */
    std::vector<cv::Point2f> pockets = { {50, 50}, {minimap.cols / 2, 50}, {750, 50},
                                         {750, 350}, {minimap.cols / 2, 350}, {50, 350} };

    cv::VideoWriter videoWriter("../output/minimap_output.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 30, cv::Size(800, 400));

    cv::Mat frame;
    cap.read(frame);
    /* Structures to save relevant information of each ball. */
    std::vector<Ball2D> balls;
    for (const auto& box : bboxes_)
    {
        Ball2D ball;
        ball.id = box.ID;
        ball.bbox = cv::Rect(box.x, box.y, box.width, box.height);
        ball.tracker = cv::TrackerCSRT::create();
        ball.trajectory = {};
        balls.push_back(ball);
    }
    /* Initialize a tracker for each ball. */
    for (auto& ball : balls)
    {
        ball.tracker->init(frame, ball.bbox);
    }

    while (cap.read(frame))
    {
        std::vector<cv::Point2f> ballPositions;
        /* Update the tracker, and balls' position. */
        for (auto& ball : balls)
        {
            ball.tracker->update(frame, ball.bbox);
            cv::Point2f center(ball.bbox.x + ball.bbox.width / 2.0f, ball.bbox.y + ball.bbox.height / 2.0f);
            ballPositions.push_back(center);
        }
        /* Compute positions of the balls in the minimap, using homography. */
        std::vector<cv::Point2f> topViewPositions;
        cv::perspectiveTransform(ballPositions, topViewPositions, homography);
        /* Update trajectories. */
        for (size_t i = 0; i < balls.size(); ++i)
        {
            balls[i].trajectory.push_back(topViewPositions[i]);
        }
        /* Draw the minimap. */
        update2DMap(minimap, balls, pockets);

        // cv::imshow("Original Frame", frame);
        cv::imshow("2D Map", minimap);

        videoWriter.write(minimap);

        /* Press 'q' to quit. */
        if (cv::waitKey(30) == 'q') break;
    }

    cv::imwrite("../output/video_last_frame.png", minimap);
    videoWriter.release();
    cap.release();

    return true;
}


BallsTracker::BallsTracker(const std::vector<Utils::BallBoundingBox>& boxes, const std::vector<Utils::Corner>& corners, std::string outputFolder)
{
    for (auto& box : boxes)
    {
        bboxes_.push_back(box);
    }
    for (auto& corner : corners)
    {
        corners_.push_back(cv::Point2f(corner.point.x, corner.point.y));
    }
    outputFolder_ = outputFolder;
}