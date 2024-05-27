#include "flyappy_autonomy_code/utils/conversion.hpp"

namespace Conversions
{    
    // Convert from Eigen to OpenCV point
    template <class T>
    cv::Point_<T> convertEigenToCVPoint(const Eigen::Matrix<T, 2, 1>& vec)
    {
        return cv::Point_<T>(vec.x(), vec.y());
    }
    template cv::Point2f convertEigenToCVPoint<float>(const Eigen::Vector2f& vec);
    template cv::Point2d convertEigenToCVPoint<double>(const Eigen::Vector2d& vec);

    // Convert from OpenCV point to Eigen
    template <class T>
    Eigen::Matrix<T, 2, 1> convertCVPointToEigen(const cv::Point_<T>& vec)
    {
        return Eigen::Matrix<T, 2, 1>(vec.x, vec.y);
    }
    template Eigen::Vector2f convertCVPointToEigen<float>(const cv::Point2f& vec);
    template Eigen::Vector2d convertCVPointToEigen<double>(const cv::Point2d& vec);
} // namespace Conversions
