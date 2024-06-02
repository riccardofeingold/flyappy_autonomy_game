#include "flyappy_autonomy_code/flyappy_ros.hpp"

constexpr uint32_t QUEUE_SIZE = 5u;

FlyappyRos::FlyappyRos(ros::NodeHandle& nh)
    : pub_acc_cmd_(nh.advertise<geometry_msgs::Vector3>("/flyappy_acc", QUEUE_SIZE)),
      pub_pos_(nh.advertise<geometry_msgs::Vector3>("/flyappy_pos", QUEUE_SIZE)),
      sub_vel_(nh.subscribe("/flyappy_vel", QUEUE_SIZE, &FlyappyRos::velocityCallback,
                            this)),
      sub_laser_scan_(nh.subscribe("/flyappy_laser_scan", QUEUE_SIZE,
                                   &FlyappyRos::laserScanCallback, this)),
      sub_game_ended_(nh.subscribe("/flyappy_game_ended", QUEUE_SIZE,
                                   &FlyappyRos::gameEndedCallback, this))
{
    flyappy_ = std::make_unique<Flyappy>();
}

void FlyappyRos::velocityCallback(const geometry_msgs::Vector3::ConstPtr& msg)
{
    flyappy_->storeObservations(Eigen::Vector2f(msg->x, msg->y));
}

void FlyappyRos::laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    flyappy_->storeObservations(msg);
    flyappy_->update();
    Eigen::Vector2d accel = flyappy_->getControlInput();

    // Send Control Input to Flyappy
    geometry_msgs::Vector3 acc_cmd;

    acc_cmd.x = accel[0];
    acc_cmd.y = accel[1];
    pub_acc_cmd_.publish(acc_cmd);

    // Publish Current Position
    geometry_msgs::Vector3 pos;
    Eigen::Vector4d curr_pos = flyappy_->XRef_;
    pos.x = curr_pos[0];
    pos.y = curr_pos[2];

    pub_pos_.publish(pos);
}

void FlyappyRos::gameEndedCallback(const std_msgs::Bool::ConstPtr& msg)
{
    if (msg->data)
    {
        ROS_INFO("Crash detected.");
    }
    else
    {
        ROS_INFO("End of countdown.");
    }

    // resetting agent
    flyappy_.reset();
    flyappy_ = std::make_unique<Flyappy>();
}
