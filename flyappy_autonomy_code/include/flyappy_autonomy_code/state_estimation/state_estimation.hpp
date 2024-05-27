#pragma once

#include <Eigen/Core>
#include <iostream>

class StateEstimation
{
    public:
    StateEstimation(const double sampling_time);

    // update state
    void update(const Eigen::Vector2f& vel_measured);

    // setters
    void setPosition(const Eigen::Vector2f& pos);

    // getters
    Eigen::Vector2f getPosition();
    Eigen::Vector2f getVelocity();

    private:
    Eigen::Vector2f position_;
    Eigen::Vector2f velocity_;
    const double sampling_time_;
};