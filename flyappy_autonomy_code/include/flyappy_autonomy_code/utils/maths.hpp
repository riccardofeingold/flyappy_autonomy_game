#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>

namespace Maths
{
    /// @brief returns the point that has the biggest x-component
    /// @param vecList contains the pointcloud
    Eigen::Vector2f farthestPoint(const std::vector<Eigen::Vector2f>& vecList);

    /// @brief returns the point which the closest to the bird (only looking at x-component)
    /// @param vecList contains pointcloud
    Eigen::Vector2f closestPoint(const std::vector<Eigen::Vector2f>& vecList);
    Eigen::Vector2f closestPoint(const std::vector<cv::Point2f>& vecList);
    
    /// @brief computes the centroid of a cluster
    /// @param points contains all the points that belong to a cluster
    /// @return the centroid of a cluster
    Eigen::Vector2f mean(const std::vector<cv::Point2f>& points);
} // namespace Maths
