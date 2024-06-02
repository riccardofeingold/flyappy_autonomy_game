#pragma once

#include <sensor_msgs/LaserScan.h>

#include "flyappy_autonomy_code/controller/pid_controller.hpp"
#include "flyappy_autonomy_code/perception/gate_detection.hpp"
#include "flyappy_autonomy_code/controller/lqr_controller.hpp"
#include "flyappy_autonomy_code/controller/mpc_controller.hpp"
#include "flyappy_autonomy_code/state_estimation/state_estimation.hpp"

// Utils
#include "flyappy_autonomy_code/utils/constants.hpp"

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
  Eigen::Vector2d getControlInput();

  private:
  /// @brief Set point: represents the position of the gate
  Eigen::Vector4d XRef_;
  /// @brief to safe gate 1 position => ensures independence of changes in gate detector class
  Eigen::Vector2d gatePosition_;
  
  /// @brief PID controller
  PIDController pid_;
  /// @brief LQR controller
  LQR lqr_;
  /// @brief MPC controller
  MPCController mpc_;

  /// @brief for state estimation
  std::shared_ptr<StateEstimation> stateEstimator_;

  /// @brief Used for gate detection
  GateDetection gateDetector_;


  /// @brief Saves the current state in the state machine
  States currentState_ = States::INIT;
  /// @brief Is true if the state has changed in the state machine
  bool stateChanged = false;
  
  /// @brief Measured velocity
  Eigen::Vector2f velMeasured_;
  /// @brief Measured LiDAR
  sensor_msgs::LaserScan laserData_;
  
  /// @brief used to store the control input returned by a controller
  Eigen::Vector2d controlInput_ = Eigen::Vector2d::Zero();
};
