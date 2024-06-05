#pragma once

#include <Eigen/Core>
#include <iostream>

class StateEstimation
{
    public:
    StateEstimation(const double sampling_time);

    /// @brief update state
    /// @param vel_measured
    void update(const Eigen::Vector2f& vel_measured);

    // setters
    /// @brief lets you set the position
    /// @param pos the position of the bird
    void setPosition(const Eigen::Vector2f& pos);

    // getters
    /// @brief returns the state vector x = [x, x_vel, y, y_vel]
    Eigen::Vector4f getStateVector();
    /// @brief returns the position vector 
    Eigen::Vector2f getPosition();
    /// @brief returns the velocity vector
    Eigen::Vector2f getVelocity();

    private:
    Eigen::Vector2f position_;
    Eigen::Vector2f velocity_;
    const double sampling_time_;
};