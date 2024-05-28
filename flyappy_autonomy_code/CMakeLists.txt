cmake_minimum_required(VERSION 3.16.3)
project(flyappy_autonomy_code)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CXX_EXTENSIONS false)

include(CTest)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

catkin_package()

set(qpOASES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../qpOASES")
set(qpOASES_LIB_DIR "${qpOASES_DIR}/bin")
set(qpOASES_INCLUDE_DIR "${qpOASES_DIR}/include")

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${qpOASES_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(flyappy
  # agent
  src/agent/flyappy.cpp
  src/agent/system_dynamics.cpp
  # controllers
  src/controller/lqr_controller.cpp
  src/controller/mpc_controller.cpp
  src/controller/pid_controller.cpp
  # state estimation
  src/state_estimation/state_estimation.cpp
  # perception
  src/perception/gate_detection.cpp
  # utils
  src/utils/conversion.cpp
  src/utils/maths.cpp
)

add_library(flyappy_ros 
  src/flyappy_ros.cpp
)

target_include_directories(flyappy_ros PRIVATE ${catkin_INCLUDE_DIRS} ${qpOASES_INCLUDE_DIR})
target_link_libraries(flyappy_ros
  flyappy
  ${catkin_LIBRARIES}
  ${qpOASES_LIB_DIR}/libqpOASES.so
  ${OpenCV_LIBS}
)

add_executable(flyappy_autonomy_code_node src/flyappy_autonomy_code_node.cpp)
target_include_directories(flyappy_autonomy_code_node PRIVATE ${catkin_INCLUDE_DIRS} ${qpOASES_INCLUDE_DIR})
target_link_libraries(flyappy_autonomy_code_node
  flyappy_ros
  ${catkin_LIBRARIES}
  ${qpOASES_LIB_DIR}/libqpOASES.so
  ${OpenCV_LIBS}
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
