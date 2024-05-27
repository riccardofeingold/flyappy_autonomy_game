#include <flyappy_autonomy_code/controller/pid_controller.hpp>

PIDController::PIDController(
    float kp,
    float ki,
    float kd,
    float i_max,
    float i_min
) : kp_(kp), 
    ki_(ki),
    kd_(kd),
    i_max_(i_max_),
    i_min_(i_min_)
{
    error_sum_ = Eigen::Vector2f::Zero();
}

Eigen::Vector2f PIDController::computeAcceleration(const Eigen::Vector2f& ref_pos, const Eigen::Vector2f& ref_vel, const Eigen::Vector2f& current_pos, const Eigen::Vector2f& current_vel)
{
    Eigen::Vector2f error_pos = ref_pos - current_pos;
    Eigen::Vector2f error_vel = ref_vel - current_vel;

    error_sum_ += error_pos;
    error_sum_[0] = error_sum_[0] > i_max_ ? i_max_ : error_sum_[0];
    error_sum_[0] = error_sum_[0] < i_min_ ? i_min_ : error_sum_[0];
    error_sum_[1] = error_sum_[1] > i_max_ ? i_max_ : error_pos[1];
    error_sum_[1] = error_sum_[1] < i_min_ ? i_min_ : error_pos[1];

    return kp_ * error_pos + ki_ * error_sum_ + kd_ * error_vel;
}