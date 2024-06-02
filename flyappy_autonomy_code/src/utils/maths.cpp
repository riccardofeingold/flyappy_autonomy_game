#include "flyappy_autonomy_code/utils/maths.hpp"
namespace Maths
{
    Eigen::Vector2f farthestPoint(const std::vector<Eigen::Vector2f>& vecList)
    {
        auto maxElement = std::max_element(vecList.begin(), vecList.end(), [](const Eigen::Vector2f& a, const Eigen::Vector2f& b){
            return a.x() < b.x();
        });

        if (maxElement == vecList.end())
        {
            return Eigen::Vector2f::Zero();
        }
        
        return *maxElement;
    }

    Eigen::Vector2f closestPoint(const std::vector<Eigen::Vector2f>& vecList)
    {
        auto minElement = std::min_element(vecList.begin(), vecList.end(), [](const Eigen::Vector2f& a, const Eigen::Vector2f& b){
            return a.x() < b.x();
        });

        if (minElement == vecList.end())
            return Eigen::Vector2f::Zero();

        return *minElement;
    }

    Eigen::Vector2f closestPoint(const std::vector<cv::Point2f>& vecList)
    {
        auto minElement = std::min_element(vecList.begin(), vecList.end(), [](const cv::Point2f& a, const cv::Point2f& b){
            return a.x < b.x;
        });

        return Eigen::Vector2f(minElement->x, minElement->y);
    }

    Eigen::Vector2f mean(const std::vector<cv::Point2f>& points)
    {
        Eigen::Vector2f meanPoint(0, 0);
        
        for (const auto& point : points)
        {
            meanPoint[0] += point.x;
            meanPoint[1] += point.y;
        }

        meanPoint /= points.size();

        return meanPoint;
    }
} // namespace Maths