#pragma once

// ROS
#include "sensor_msgs/LaserScan.h"

// Libraries
#include <unordered_set>
#include <cmath>
#include <Eigen/Core>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Utils
#include "flyappy_autonomy_code/utils/conversion.hpp"
#include "flyappy_autonomy_code/utils/constants.hpp"
#include "flyappy_autonomy_code/utils/maths.hpp"

#define DEBUG false

struct PointHasher
{
    std::size_t operator()(const Eigen::Vector2f& point) const
    {
        return std::hash<float>()(point.x()) ^ std::hash<float>()(point.y());
    }
};

struct PointGroup 
{
    cv::Point2f centroid;
    std::vector<cv::Point2f> points;
};

struct Gate
{
    bool gateUpdated = false;
    float upperBoundY;
    float lowerBoundY;
    Eigen::Vector2f position = Eigen::Vector2f::Zero();
    Eigen::Vector2f prevPosition = Eigen::Vector2f::Zero();
};

enum ResetStates
{
    CLEAR_ALL,
    CLEAR_OLDEST,
    CLEAR_ALL_NEAR_CLOSEST_POINT
};

class GateDetection
{
    public:
    GateDetection(
        int mapWidth = 1000,
        int mapHeight = 1000,
        int pointcloudBufferSize = 10000,
        float minGateHeight = 0.5,
        float wallWidth = 1,
        int decimation = 6,
        int numClusters = 4
    );

    /// @brief Returns two clusters after running KMeans on pointcloud data, which can be then used to compute two convex hulls
    void clustering(std::vector<PointGroup>& clusters);

    /// @brief Returns convex hulls based on a set of 2D points
    /// @param clusters should be a list returned from clustering()
    /// @param enoughDataPoints is returned true if dataPoints has at least 3 points to form a convex hull
    void convexHull(const std::vector<PointGroup>& clusters, std::vector<PointGroup>& hulls, bool& enoughDataPoints);

    /// @brief returns the center position of the gate, including the upper and lower bound
    /// @param hulls Convex hulls that represent the upper and lower pipes
    void getGatePosition(const std::vector<PointGroup>& hulls);

    /// @brief Main method: responsible for checking possible gates, and updating the pointcloud
    /// @param position Current Position of flyappy bird
    /// @param laserData current laser measurements
    void update(const Eigen::Vector2f& position, const sensor_msgs::LaserScan& laserData);

    /// @brief Clears pointCloud2D_
    void reset(ResetStates state = ResetStates::CLEAR_ALL);

    /// @brief Sets the upper and lower boundaries as soon as we get a measurement from the top and the bottom laser
    void computeBoundaries();

    /// @brief Returns the relative position of a measured distance by a laser w.r.t. flyappy bird
    /// @param index Specifies the index used to get the distance from ROS Laser message
    /// @param distance measured distance by laser at index
    /// @param angleIncrement angle between two consecutive lasers
    Eigen::Vector2f computeRelativePointPosition(const int index, const double distance, const double angleIncrement);
    
    /// @brief Becomes true if the upper and lower boundary are set (used for filtering point cloud)
    bool initDone = false;

    /// @brief contains position and upper and lower bound of detected gate
    std::unique_ptr<Gate> gate;

    /// @brief closest points of to flyappy bird
    Eigen::Vector2f closestPoint = Eigen::Vector2f::Zero();
    
    private:
    /// @brief renders pointcloud using OpenCV
    void renderMap();

    /// @brief renders clusters and convex hulls using OpenCV
    /// @param clusters obtained from clustering
    /// @param hulls obtained from convexHull()
    void renderMap(const std::vector<PointGroup>& clusters, const std::vector<PointGroup>& hulls);

    /// @brief renders clusters and convex hulls using OpenCV
    /// @param clusters obtained from clustering
    void renderMap(const std::vector<PointGroup>& clusters);

    /// @brief returns true if point is not part of ground or ceiling
    /// @param point Is the point to check for feasibility.
    bool feasible(const Eigen::Vector2f& point);

    /// @brief filter out duplicate points
    /// @param dataPoints
    /// @param threshold defines how close two points have to be in order to be identical
    std::vector<Eigen::Vector2f> filterDuplicatePoints(const std::vector<Eigen::Vector2f>& dataPoints, float threshold);

    /// @brief stores all 2D points computed from the laser data
    std::vector<Eigen::Vector2f> pointCloud2D_;

    // Laser Data Processing Variables
    /// @brief counts how many times flappy bird goes out of screen in pointcloud window
    int countPointcloudWindowUpdates_ = 0;
    /// @brief Margin applied to top and bottom to filter out points from ceiling and ground
    const float margin_ = 10 * pixelInMeters;
    /// @brief upper margin to filter out points that are part of the ceiling
    float upperBoundary_ = 0;
    /// @brief lower margin to filter out points that are part of the ground
    float lowerBoundary_ = 0;

    /// @brief to keep track of update iterations
    int updateIterations_ = 0;

    // Settings
    int numClusters_;
    const int decimation_;
    const int mapWidth_;
    const int mapHeight_;
    const int pointcloudBufferSize_;
    const float minGateHeight_;
    const float wallWidth_;
};