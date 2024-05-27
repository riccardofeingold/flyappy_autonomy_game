#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace Conversions
{
    template <class T>
    cv::Point_<T> convertEigenToCVPoint(const Eigen::Matrix<T, 2, 1>& vec);

    template <class T>
    Eigen::Matrix<T, 2, 1> convertCVPointToEigen(const cv::Point_<T>& vec);
} // Namespace Conversions