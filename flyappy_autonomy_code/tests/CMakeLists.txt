find_package(GTest REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include(GoogleTest)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(test_my_feature test_my_feature.cpp)
target_link_libraries(test_my_feature
  flyappy
  gtest
  gtest_main
  ${OpenCV_LIBS}
)
gtest_discover_tests(test_my_feature)
