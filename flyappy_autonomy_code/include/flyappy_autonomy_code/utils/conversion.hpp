#pragma once

#include <Eigen/Core>
#include <qpOASES.hpp>
#include <opencv2/opencv.hpp>

USING_NAMESPACE_QPOASES
namespace Conversions
{
    /// @brief Convert from Eigen to OpenCV point
    template <class T>
    cv::Point_<T> convertEigenToCVPoint(const Eigen::Matrix<T, 2, 1>& vec);

    /// @brief Convert from OpenCV point to Eigen
    template <class T>
    Eigen::Matrix<T, 2, 1> convertCVPointToEigen(const cv::Point_<T>& vec);

    /// @brief convert eigen matrix to real_t (qpOASES format)
    template <class T>
    real_t* convertEigenToRealT(const Eigen::Matrix<T, -1, -1>& eigenMatrix);
} // Namespace Conversions