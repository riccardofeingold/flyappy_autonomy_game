#include "flyappy_autonomy_code/state_estimation/state_estimation.hpp"

StateEstimation::StateEstimation(const double sampling_time) : 
    sampling_time_(sampling_time)
{
    position_ = Eigen::Vector2f(0.0, 0.0);
    velocity_ = Eigen::Vector2f(0.0, 0.0);
}

// update state
void StateEstimation::update(const Eigen::Vector2f& vel_measured)
{
    velocity_ = vel_measured;
    position_ += vel_measured * sampling_time_;
}

// Setters
void StateEstimation::setPosition(const Eigen::Vector2f& pos) { position_ = pos; }

// Getters
Eigen::Vector2f StateEstimation::getPosition() { return position_; }
Eigen::Vector2f StateEstimation::getVelocity() { return velocity_; }