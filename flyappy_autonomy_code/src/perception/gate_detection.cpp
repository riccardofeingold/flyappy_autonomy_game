#include "flyappy_autonomy_code/perception/gate_detection.hpp"

GateDetection::GateDetection(
    int mapWidth, 
    int mapHeight,
    int pointcloudBufferSize,
    int numClusters,
    bool renderMap
) : mapWidth_(mapWidth), 
    mapHeight_(mapHeight),
    pointcloudBufferSize_(pointcloudBufferSize),
    renderMap_(renderMap)
{
    numClusters_ = numClusters;
    gate1 = std::make_unique<Gate>();
}


void GateDetection::update(const Eigen::Vector2f& position, const sensor_msgs::LaserScan& laserData, const States& currentState)
{
    // reset based on travelled distance
    if (pointCloud2D_.size() > pointcloudBufferSize_)
    {
        std::cout << "CLEAR OLDEST" << std::endl;
        reset(ResetStates::CLEAR_OLDEST);
    } else if (position.x() > closestPoints.closestPointWall1.x() && currentState == States::TUNNEL)
    {
        std::cout << "CLEAR ALL NEAR CLOSEST POINT" << std::endl;
        
        gate1->prevPosition = gate1->position;
        reset(ResetStates::CLEAR_ALL_NEAR_CLOSEST_POINT);
    }


    // Save usefull 2D points
    for (uint8_t index = 0; index < laserData.intensities.size(); ++index)
    {
        if (laserData.intensities[index] == 1)
        {
            Eigen::Vector2f point = position + computeRelativePointPosition(index, laserData.ranges[index], laserData.angle_increment);

            if (currentState != States::INIT)
            {
                if (feasible(point))
                {
                    pointCloud2D_.push_back(point);
                }
            } else
            {
                pointCloud2D_.push_back(point);
            }
        }
    }

    // perform clustering and obtain convex hulls
    if (pointCloud2D_.size() > 0)
    {
        // filter duplicates
        pointCloud2D_ = filterDuplicatePoints(pointCloud2D_, 0.0001);

        // check if we still have data points left
        if (pointCloud2D_.size() == 0)
        {
            std::cout << "FILTER REMOVED ALL DATA POINTS" << std::endl;
            return;
        }

        // find closest point
        if (pointCloud2D_.size() > 0)
        {
            closestPoints.closestPointWall1 = Maths::closestPoint(pointCloud2D_);
            gate1->position[0] = closestPoints.closestPointWall1[0];
        }
    } 
    
    if (renderMap_)
    {
        // Refresh window
        Eigen::Vector2f maxPoint = Maths::farthestPoint(pointCloud2D_);
        if (maxPoint.x() * (int) 1 /pixelInMeters > mapWidth_ * (countPointcloudWindowUpdates_ + 1))
        {
            countPointcloudWindowUpdates_++;
        }
        renderMap(position);
    }
}

void GateDetection::reset(ResetStates state)
{
    #if DEBUG
    for (std::vector<Eigen::Vector2f>::iterator it = pointCloud2D_.begin(); it != pointCloud2D_.end(); ++it)
    {
        std::cout << it->x() << "," << it->y() << "," << "\n"; 
    }
    #endif

    switch(state)
    {
        case ResetStates::CLEAR_ALL:
            pointCloud2D_.clear();
            break;
        
        case ResetStates::CLEAR_OLDEST:
            pointCloud2D_.erase(pointCloud2D_.begin(), pointCloud2D_.begin() + static_cast<int>(pointCloud2D_.size() * 0.75));
            break;

        case ResetStates::CLEAR_ALL_NEAR_CLOSEST_POINT:
            std::vector<Eigen::Vector2f> temp;
            for (unsigned int i = 0; i < pointCloud2D_.size(); ++i)
            {
                if (pointCloud2D_[i].x() > closestPoints.closestPointWall1.x() + 2*wallWidth)
                {
                    temp.push_back(pointCloud2D_[i]);
                }
            }
            pointCloud2D_ = temp;
            break;
    }
}

void GateDetection::sortPointCloud(std::vector<Eigen::Vector2f>& pointcloud)
{
    std::sort(pointcloud.begin(), pointcloud.end(), [](const Eigen::Vector2f& a, const Eigen::Vector2f& b){
        return a.y() > b.y();
    });
}

void GateDetection::findGapInWall(const Eigen::Vector2f& position)
{
    // convert from Eigen to OpenCV
    std::vector<Eigen::Vector2f> pointCloudCVWall1;
    for (auto point : pointCloud2D_)
    {
        if (point.x() >= closestPoints.closestPointWall1.x() && point.x() < closestPoints.closestPointWall1.x() + wallWidth)
        {
            pointCloudCVWall1.push_back(point);
        }
    }

    // only continue if we have at least 10 data points
    if (pointCloudCVWall1.size() < 10) return;

    // sort pointclouds
    sortPointCloud(pointCloudCVWall1);

    gate1->position[0] = closestPoints.closestPointWall1.x();

    float maxGap = 0.0f;
    float max2ndGap = 0.0f;
    int indexBiggestGap = 0;
    int index2ndBiggestGap = 0;
    for (int i = 0; i < pointCloudCVWall1.size() - 1; ++i)
    {
        float gap = pointCloudCVWall1[i].y() - pointCloudCVWall1[i+1].y();
        if (gap > maxGap)
        {
            max2ndGap = maxGap;
            index2ndBiggestGap = indexBiggestGap;
            
            maxGap = gap;
            indexBiggestGap = i;
        }
    }

    float yPos1 = (pointCloudCVWall1[indexBiggestGap].y() + pointCloudCVWall1[indexBiggestGap + 1].y()) / 2.0;
    float yPos2 = (pointCloudCVWall1[index2ndBiggestGap].y() + pointCloudCVWall1[index2ndBiggestGap + 1].y()) / 2.0;

    float dist1 = std::abs(yPos1 - position.y());
    float dist2 = std::abs(yPos2 - position.y());

    if (dist1 < dist2 || max2ndGap < 2*birdHeight)
    {
        gate1->upperBoundY = pointCloudCVWall1[indexBiggestGap].y();
        gate1->lowerBoundY = pointCloudCVWall1[indexBiggestGap + 1].y();
        gate1->position[1] = yPos1;
    } else
    {
        gate1->upperBoundY = pointCloudCVWall1[index2ndBiggestGap].y();
        gate1->lowerBoundY = pointCloudCVWall1[index2ndBiggestGap + 1].y();
        gate1->position[1] = yPos2;
    }
}

bool GateDetection::clustering(std::vector<PointGroup>& clustersWall1)
{
    // convert from Eigen to OpenCV
    std::vector<cv::Point2f> pointCloudCVWall1;
    std::vector<cv::Point2f> pointCloudCVWall2;
    for (auto point : pointCloud2D_)
    {
        if (point.x() >= closestPoints.closestPointWall1.x() && point.x() < closestPoints.closestPointWall1.x() + wallWidth)
        {
            pointCloudCVWall1.push_back(Conversions::convertEigenToCVPoint<float>(point));
        } else
        {
            pointCloudCVWall2.push_back(Conversions::convertEigenToCVPoint<float>(point));
        }
    }

    // get closes point of the second wall
    if (pointCloudCVWall2.size() > 0)
    {
        this->closestPoints.closestPointWall2 = Maths::closestPoint(pointCloudCVWall2);
    }

    // Check if we have more data poitns than clusters
    if (pointCloudCVWall1.size() < clustersWall1.size())
    {
        std::cout << "WALL: NOT ENOUGH DATA POINTS NEAR CLOSEST POINT" << std::endl;
        return false;
    }

    // WALL
    // convert points to Mat
    cv::Mat pointsMatWall1(pointCloudCVWall1.size(), 1, CV_32FC2, &pointCloudCVWall1[0]);

    // Perform K-Means clustering
    cv::Mat labelsWall1, centersWall1;
    cv::kmeans(pointsMatWall1, clustersWall1.size(), labelsWall1,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               10, cv::KMEANS_PP_CENTERS, centersWall1);

    // Separate points into clusters
    for (int i = 0; i < pointCloudCVWall1.size(); ++i) {
        int clusterIdx = labelsWall1.at<int>(i);
        clustersWall1[clusterIdx].points.push_back(pointCloudCVWall1[i]);
    }

    // compute centroids
    for (auto& cluster : clustersWall1)
    {
        cluster.centroid = Conversions::convertEigenToCVPoint<float>(Maths::mean(cluster.points));
    }

    // order clusters according to y-centroid from top to bottom
    std::sort(clustersWall1.begin(), clustersWall1.end(), [](const PointGroup& a, const PointGroup& b) -> bool {
        return a.centroid.y < b.centroid.y;
    });

    return true;
}

void GateDetection::convexHull(const std::vector<PointGroup>& clusters, std::vector<PointGroup>& hulls, bool& enoughDataPoints)
{
    for (uint8_t i = 0; i < clusters.size(); ++i)
    {
        if (clusters[i].points.size() >= 3)
        {
            cv::convexHull(clusters[i].points, hulls[i].points);
            hulls[i].centroid = clusters[i].centroid;

            enoughDataPoints = true;
        } else
        {
            enoughDataPoints = false;
            break;
        }
    }
}

bool GateDetection::feasible(const Eigen::Vector2f& point)
{
    if (point.y() < upperBoundary_ - margin_ && point.y() > lowerBoundary_ + margin_)
    {
        return true;
    }

    return false;
}

std::vector<Eigen::Vector2f> GateDetection::filterDuplicatePoints(const std::vector<Eigen::Vector2f>& dataPoints, float threshold)
{
    std::unordered_set<Eigen::Vector2f, PointHasher> uniquePoints;
    std::vector<Eigen::Vector2f> filteredPoints;

    for (const auto& point : dataPoints)
    {
        bool isUnique = true;
        for (const auto& uniquePoint : uniquePoints)
        {
            if (std::abs(uniquePoint.x() - point.x()) < threshold || std::abs(uniquePoint.y() - point.y()) < threshold)
            {
                isUnique = false;
                break;
            }
        }

        if (isUnique)
        {
            uniquePoints.insert(point);
            filteredPoints.push_back(point);
        }
    }

    return filteredPoints;
}

void GateDetection::getGatePosition(const std::vector<PointGroup>& hullsWall1, const Eigen::Vector2f& position)
{
    float maxGapWall1 = 0;
    float maxGapWall2 = 0;
    for (size_t i = 0; i < hullsWall1.size() - 1; ++i)
    {
        float min1 = upperBoundary_;
        float max1 = lowerBoundary_;
        float min2 = upperBoundary_;
        float max2 = lowerBoundary_;

        for (const auto& point1 : hullsWall1[i].points)
        {
            if (point1.y < min1) min1 = point1.y;
            if (point1.y > max1) max1 = point1.y;
        }

        for (const auto& point2 : hullsWall1[i + 1].points)
        {
            if (point2.y < min2) min2 = point2.y;
            if (point2.y > max2) max2 = point2.y;
        }

        float gap = std::max(min1, min2) - std::min(max1, max2);
        if (gap > maxGapWall1)
        {
            maxGapWall1 = gap;
            gate1->lowerBoundY = std::min(max1, max2);
            gate1->upperBoundY = std::max(min1, min2);
        }
    }
    
    float gap1 = std::abs(gate1->upperBoundY - gate1->lowerBoundY);

    std::cout << "GAP1: " << gap1 << std::endl;

    gate1->position[0] = closestPoints.closestPointWall1.x();

    if (gap1 > 2*birdHeight && gap1 < gateHeight + 0.2)
    {
        std::cout << "USING BIGGEST" << std::endl;
        gate1->position[1] = (gate1->upperBoundY + gate1->lowerBoundY)/2.0f;
    }
    else
    {
        findGapInWall(position);
        std::cout << "FIND GAP IN WALL" << std::endl;
    }
}

Eigen::Vector2f GateDetection::computeRelativePointPosition(const int index, const double distance, const double angleIncrement)
{
    double angle = angleIncrement * (index - 4);
    return Eigen::Vector2f(distance * cos(angle), distance * sin(angle));
}

void GateDetection::computeBoundaries()
{
    auto highestPoint = std::max_element(pointCloud2D_.begin(), pointCloud2D_.end(), [](const Eigen::Vector2f& a, const Eigen::Vector2f& b){
        return a.y() < b.y();
    });
    
    auto lowestPoint = std::min_element(pointCloud2D_.begin(), pointCloud2D_.end(), [](const Eigen::Vector2f& a, const Eigen::Vector2f& b){
        return a.y() < b.y();
    });

    upperBoundary_ = highestPoint->y();
    lowerBoundary_ = lowestPoint->y();

    gate1->upperBoundY = upperBoundary_;
    gate1->lowerBoundY = lowerBoundary_;
}

/*********************************************************/
/**********************RENDERING**************************/
/*********************************************************/
void GateDetection::renderMap(const Eigen::Vector2f& position)
{
    cv::Mat image = cv::Mat::zeros(mapHeight_, mapWidth_, CV_8UC3);

    for (const auto& point : pointCloud2D_)
    {
        cv::Point2f cvPoint = Conversions::convertEigenToCVPoint<float>(point);
        cvPoint = cv::Point2f(cvPoint.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPoint.y);
        cv::circle(image, cvPoint * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(0, 255, 0), cv::FILLED);
    }

    // gate position
    cv::Point2f cvPointGate1 = Conversions::convertEigenToCVPoint<float>(gate1->position);
    cvPointGate1 = cv::Point2f(cvPointGate1.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointGate1.y);
    cv::circle(image, cvPointGate1 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(0, 0, 255), cv::FILLED);

    // closest point
    cv::Point2f cvPointClosestW1 = Conversions::convertEigenToCVPoint<float>(closestPoints.closestPointWall1);
    cvPointClosestW1 = cv::Point2f(cvPointClosestW1.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointClosestW1.y);
    cv::circle(image, cvPointClosestW1 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(255, 0, 0), cv::FILLED);

    // closest point
    cv::Point2f cvPointClosestW2 = Conversions::convertEigenToCVPoint<float>(closestPoints.closestPointWall2);
    cvPointClosestW2 = cv::Point2f(cvPointClosestW2.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointClosestW2.y);
    cv::circle(image, cvPointClosestW2 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(255, 0, 0), cv::FILLED);

    // closest point
    cv::Point2f bird = Conversions::convertEigenToCVPoint<float>(position);
    bird = cv::Point2f(bird.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -bird.y);
    cv::circle(image, bird * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 10, cv::Scalar(255, 10, 50), cv::FILLED);

    cv::imshow("Point Cloud", image);
    cv::waitKey(1);
}

void GateDetection::renderMap(const std::vector<PointGroup>& clusters, const std::vector<PointGroup>& hulls)
{
    cv::Mat image = cv::Mat::zeros(mapHeight_, mapWidth_, CV_8UC3);
    cv::Scalar color = cv::Scalar(0, 255, 0);

    for (uint8_t i = 0; i < clusters.size(); ++i)
    {
        if (i < hulls.size())
        {
            for (uint16_t ii = 0; ii < hulls[i].points.size(); ++ii)
            {
                cv::Point2f startPoint = cv::Point2f(hulls[i].points[ii].x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -hulls[i].points[ii].y);
                cv::Point2f endPoint = cv::Point2f(hulls[i].points[(ii + 1) % hulls[i].points.size()].x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -hulls[i].points[(ii + 1) % hulls[i].points.size()].y);
                cv::line(image, startPoint * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), endPoint * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), color, 2);
            }
        }

        for (const auto& point : clusters[i].points)
        {
            cv::Point2f p(point.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -point.y);
            cv::circle(image, p * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, color, cv::FILLED);
        }
    }
    
    // gate position
    cv::Point2f cvPointGate1 = Conversions::convertEigenToCVPoint<float>(gate1->position);
    cvPointGate1 = cv::Point2f(cvPointGate1.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointGate1.y);
    cv::circle(image, cvPointGate1 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(0, 0, 255), cv::FILLED);

    // closest point
    cv::Point2f cvPointClosestW1 = Conversions::convertEigenToCVPoint<float>(closestPoints.closestPointWall1);
    cvPointClosestW1 = cv::Point2f(cvPointClosestW1.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointClosestW1.y);
    cv::circle(image, cvPointClosestW1 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(255, 0, 0), cv::FILLED);

    // closest point
    cv::Point2f cvPointClosestW2 = Conversions::convertEigenToCVPoint<float>(closestPoints.closestPointWall2);
    cvPointClosestW2 = cv::Point2f(cvPointClosestW2.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointClosestW2.y);
    cv::circle(image, cvPointClosestW2 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(255, 0, 0), cv::FILLED);
    
    cv::imshow("Point Cloud", image);
    cv::waitKey(1);
}

void GateDetection::renderMap(const std::vector<PointGroup>& clusters)
{
    cv::Mat image = cv::Mat::zeros(mapWidth_, mapHeight_, CV_8UC3);
    cv::Scalar color = cv::Scalar(0, 255, 0);

    for (uint8_t i = 0; i < clusters.size(); ++i)
    {
        for (const auto& point : clusters[i].points)
        {
            cv::Point2f p(point.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -point.y);
            cv::circle(image, p * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, color, cv::FILLED);
        }
    }
    
    // gate position
    cv::Point2f cvPointGate1 = Conversions::convertEigenToCVPoint<float>(gate1->position);
    cvPointGate1 = cv::Point2f(cvPointGate1.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointGate1.y);
    cv::circle(image, cvPointGate1 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(0, 0, 255), cv::FILLED);

    // closest point
    cv::Point2f cvPointClosestW1 = Conversions::convertEigenToCVPoint<float>(closestPoints.closestPointWall1);
    cvPointClosestW1 = cv::Point2f(cvPointClosestW1.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointClosestW1.y);
    cv::circle(image, cvPointClosestW1 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(255, 0, 0), cv::FILLED);

    // closest point
    cv::Point2f cvPointClosestW2 = Conversions::convertEigenToCVPoint<float>(closestPoints.closestPointWall2);
    cvPointClosestW2 = cv::Point2f(cvPointClosestW2.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointClosestW2.y);
    cv::circle(image, cvPointClosestW2 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(255, 0, 0), cv::FILLED);
    
    cv::imshow("Point Cloud", image);
    cv::waitKey(1);
}