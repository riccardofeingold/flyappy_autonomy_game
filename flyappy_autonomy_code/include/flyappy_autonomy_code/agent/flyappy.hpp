#pragma once

#include <sensor_msgs/LaserScan.h>

#include "flyappy_autonomy_code/controller/pid_controller.hpp"
#include "flyappy_autonomy_code/perception/gate_detection.hpp"
#include "flyappy_autonomy_code/controller/lqr_controller.hpp"
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

  /// @brief Initial steps before running update pipeline
  void init();

  /// @brief Store measurements
  void storeObservations(const Eigen::Vector2f& vel) { velMeasured_ = vel; }
  void storeObservations(const sensor_msgs::LaserScan::ConstPtr& laserData) { laserData_ = *laserData; };

  /// @brief update state, map, path, and compute control input using MPC
  void update();

  /// @brief Get control input
  Eigen::Vector2f getControlInput();

  /// @brief PID controller
  PIDController pid_;
  /// @brief LQR controller
  LQR lqr_;

  /// @brief Used for gate detection
  GateDetection gateDetector_;

  /// @brief for state estimation
  std::shared_ptr<StateEstimation> stateEstimator_;

  private:
  /// @brief Saves the current state in the state machine
  States currentState_ = States::INIT;
  /// @brief Is true if the state has changed in the state machine
  bool stateChanged = false;
  
  /// @brief Measured velocity
  Eigen::Vector2f velMeasured_;
  /// @brief Measured LiDAR
  sensor_msgs::LaserScan laserData_;
  
  /// @brief used to store the control input returned by a controller
  Eigen::Vector2f controlInput_ = Eigen::Vector2f::Zero();
};
