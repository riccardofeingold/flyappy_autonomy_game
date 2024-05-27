#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>

namespace Maths
{
    Eigen::Vector2f farthestPoint(const std::vector<Eigen::Vector2f>& vecList);
    Eigen::Vector2f closestPoint(const std::vector<Eigen::Vector2f>& vecList);
    
    Eigen::Vector2f mean(const std::vector<cv::Point2f>& points);
} // namespace Maths
