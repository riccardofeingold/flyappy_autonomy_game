#include "flyappy_autonomy_code/utils/conversion.hpp"

namespace Conversions
{
    template <class T>
    cv::Point_<T> convertEigenToCVPoint(const Eigen::Matrix<T, 2, 1>& vec)
    {
        return cv::Point_<T>(vec.x(), vec.y());
    }
    template cv::Point2f convertEigenToCVPoint<float>(const Eigen::Vector2f& vec);
    template cv::Point2d convertEigenToCVPoint<double>(const Eigen::Vector2d& vec);

    template <class T>
    Eigen::Matrix<T, 2, 1> convertCVPointToEigen(const cv::Point_<T>& vec)
    {
        return Eigen::Matrix<T, 2, 1>(vec.x, vec.y);
    }
    template Eigen::Vector2f convertCVPointToEigen<float>(const cv::Point2f& vec);
    template Eigen::Vector2d convertCVPointToEigen<double>(const cv::Point2d& vec);

    template <class T>
    real_t* convertEigenToRealT(const Eigen::Matrix<T, -1, -1>& eigenMatrix)
    {
        int numRows = eigenMatrix.rows();
        int numCols = eigenMatrix.cols();
        real_t* realMatrix = new real_t[numRows * numCols];

        for (unsigned int row = 0; row < numRows; ++row)
        {
            for (unsigned int col = 0; col < numCols; ++col)
            {
                realMatrix[row * numCols + col] = eigenMatrix(row, col);
            }
        }
        return realMatrix;
    }

    template real_t* convertEigenToRealT<float>(const Eigen::MatrixXf& eigenMatrix);
    template real_t* convertEigenToRealT<double>(const Eigen::MatrixXd& eigenMatrix);
} // namespace Conversions
