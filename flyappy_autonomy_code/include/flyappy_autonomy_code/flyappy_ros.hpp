#pragma once

#include <ros/ros.h>
#include <qpOASES.hpp>
#include <Eigen/Core>

#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/LaserScan.h>
#include <std_msgs/Bool.h>

#include "flyappy_autonomy_code/agent/flyappy.hpp"
#include "flyappy_autonomy_code/controller/pid_controller.hpp"
#include "flyappy_autonomy_code/state_estimation/state_estimation.hpp"

class FlyappyRos
{
  public:
    FlyappyRos(ros::NodeHandle& nh);

  private:
    void velocityCallback(const geometry_msgs::Vector3::ConstPtr& msg);
    void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
    void gameEndedCallback(const std_msgs::Bool::ConstPtr& msg);

    ros::Publisher pub_pos_;          ///< Publisher for current position
    ros::Publisher pub_acc_cmd_;      ///< Publisher for acceleration command
    ros::Subscriber sub_vel_;         ///< Subscriber for velocity
    ros::Subscriber sub_laser_scan_;  ///< Subscriber for laser scan
    ros::Subscriber sub_game_ended_;  ///< Subscriber for crash detection

    std::unique_ptr<Flyappy> flyappy_;  ///< ROS-free main code
};
