#pragma once

#include <Eigen/Dense>
#include "flyappy_autonomy_code/utils/constants.hpp"

class SystemDynamics
{
    public:
    SystemDynamics();

    int Nx;
    int Nu;
    int Ny;

    Eigen::Matrix<float, 4, 4> A;
    Eigen::Matrix<float, 4, 4> Ad;
    Eigen::Matrix<float, 4, 2> B;
    Eigen::Matrix<float, 4, 2> Bd;
    Eigen::Matrix<float, 2, 4> C;
    Eigen::Matrix<float, 2, 4> Cd;

    Eigen::Vector4f nextState(const Eigen::Vector2f& Uk, const Eigen::Vector4f& Xk);
    Eigen::Vector2f getOnlyPosition(const Eigen::Vector4f& Xk);
};