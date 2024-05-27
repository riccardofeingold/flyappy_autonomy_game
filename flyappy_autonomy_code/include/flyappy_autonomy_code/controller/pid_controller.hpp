#pragma once

#include <qpOASES.hpp>
#include <iostream>
#include <Eigen/Core>

class PIDController
{
    public:
    PIDController(
        float kp = 20,
        float ki = 1,
        float kd = 2,
        float i_max = 5,
        float i_min = -5
    );

    Eigen::Vector2f computeAcceleration(const Eigen::Vector2f& ref_pos, const Eigen::Vector2f& ref_vel, const Eigen::Vector2f& current_pos, const Eigen::Vector2f& current_vel);

    private:
    float kp_;
    float ki_;
    float kd_;
    float i_max_;
    float i_min_;
    Eigen::Vector2f error_sum_;
};