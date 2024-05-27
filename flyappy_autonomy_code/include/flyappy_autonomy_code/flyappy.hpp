#pragma once

#include <sensor_msgs/LaserScan.h>

#include "flyappy_autonomy_code/controller/pid_controller.hpp"
#include "flyappy_autonomy_code/perception/gate_detection.hpp"
#include "flyappy_autonomy_code/controller/mpc_controller.hpp"
#include "flyappy_autonomy_code/state_estimation/state_estimation.hpp"

// Utils
#include "flyappy_autonomy_code/utils/constants.hpp"

enum States {
  INIT,
  MOVE_FORWARD,
  FLY
};

class Flyappy
{
  public:
  Flyappy();

  // Initial steps before running update pipeline
  void init();

  // Store measurements
  void storeObservations(const Eigen::Vector2f& vel) { velMeasured_ = vel; }
  void storeObservations(const sensor_msgs::LaserScan::ConstPtr& laserData) { laserData_ = *laserData; };

  // update state, map, path, and compute control input using MPC
  void update();

  // Get control input
  Eigen::Vector2f getControlInput();

  PIDController pid_;
  GateDetection gateDetector_;
  std::shared_ptr<StateEstimation> stateEstimator_;

  private:
  States currentState_ = States::INIT;
  bool stateChanged = false;
  
  Eigen::Vector2f velMeasured_;
  sensor_msgs::LaserScan laserData_;
  
  Eigen::Vector2f controlInput_ = Eigen::Vector2f::Zero();
};
