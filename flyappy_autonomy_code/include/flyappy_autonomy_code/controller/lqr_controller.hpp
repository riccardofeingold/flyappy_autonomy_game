#pragma once

#include "flyappy_autonomy_code/agent/system_dynamics.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

class LQR
{
    public:
    LQR();
    LQR(const Eigen::Matrix4f& Q, const Eigen::Matrix2f& R, int maxIterations);
    LQR(const Eigen::Matrix4f& Q, const Eigen::Matrix2f& R, int maxIterations, const Eigen::Matrix<float, 2, 4>& K);
    
    Eigen::Vector2f eval(const Eigen::Vector4f& deltaXk);

    private:
    void computeK();

    SystemDynamics system_;

    Eigen::Matrix4f Q_ = Eigen::Matrix4f::Identity();
    Eigen::Matrix2f R_ = Eigen::Matrix2f::Identity();
    Eigen::Matrix<float, 2, 4> K_ = Eigen::Matrix<float, 2, 4>::Zero();

    int maxIterations_;

    // Friend test classes
    FRIEND_TEST(ControllerTesting, LQRTest);
};