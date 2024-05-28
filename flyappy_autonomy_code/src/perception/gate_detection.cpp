#include "flyappy_autonomy_code/perception/gate_detection.hpp"

GateDetection::GateDetection(
    int mapWidth, 
    int mapHeight,
    int pointcloudBufferSize,
    float minGateHeight,
    float wallWidth,
    int decimation,
    int numClusters
) : mapWidth_(mapWidth), 
    mapHeight_(mapHeight),
    pointcloudBufferSize_(pointcloudBufferSize),
    minGateHeight_(minGateHeight),
    wallWidth_(wallWidth),
    decimation_(decimation)
{
    numClusters_ = numClusters;
    gate1 = std::make_unique<Gate>();
    gate2 = std::make_unique<Gate>();
}

void GateDetection::update(const Eigen::Vector2f& position, const sensor_msgs::LaserScan& laserData)
{
    // reset based on travelled distance
    if (pointCloud2D_.size() > pointcloudBufferSize_)
    {
        std::cout << "CLEAR OLDEST" << std::endl;
        reset(ResetStates::CLEAR_OLDEST);
    } else if (position.x() > gate1->position.x() + wallWidth_/2)
    {
        std::cout << "CLEAR ALL NEAR CLOSEST POINT" << std::endl;
        
        gate1->prevPosition = gate1->position;
        reset(ResetStates::CLEAR_ALL_NEAR_CLOSEST_POINT);

        std::vector<Eigen::Vector2f> filteredPoints = filterDuplicatePoints(pointCloud2D_, 0.0001);
        this->closestPoints.closestPointWall1 = Maths::closestPoint(filteredPoints);
        gate1->position[0] = this->closestPoints.closestPointWall1.x();
    }

    // Refresh window
    Eigen::Vector2f maxPoint = Maths::farthestPoint(pointCloud2D_);
    if (maxPoint.x() * (int) 1 /pixelInMeters > mapWidth_ * (countPointcloudWindowUpdates_ + 1))
    {
        countPointcloudWindowUpdates_++;
    }

    // Save usefull 2D points
    for (uint8_t index = 0; index < laserData.intensities.size(); ++index)
    {
        if (laserData.intensities[index] == 1)
        {
            Eigen::Vector2f point = position + computeRelativePointPosition(index, laserData.ranges[index], laserData.angle_increment);

            if (initDone)
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

    // // perform clustering and obtain convex hulls
    if (initDone && pointCloud2D_.size() > 100 && (updateIterations_ + 1) % decimation_ == 0)
    {
        std::vector<PointGroup> clustersWall1(numClusters_);
        std::vector<PointGroup> clustersWall2(numClusters_);
        clustering(clustersWall1, clustersWall2);

        bool enoughDataPointsWall1;
        std::vector<PointGroup> hullsWall1(numClusters_);
        convexHull(clustersWall1, hullsWall1, enoughDataPointsWall1);

        if (enoughDataPointsWall1)
        {
            clustersWall1.insert(clustersWall1.end(), clustersWall2.begin(), clustersWall2.end());
            renderMap(clustersWall1, hullsWall1);
            getGatePosition(hullsWall1, clustersWall2);
        }
        else
        {
            std::cout << "NOT ENOUGH DATA POINTS FOR CONVEX HULL" << std::endl;
            getGatePosition(clustersWall1, clustersWall2);
            clustersWall1.insert(clustersWall1.end(), clustersWall2.begin(), clustersWall2.end());
            renderMap(clustersWall1);
        }

        updateIterations_ = 0;
    } else
    {
        renderMap();
    }

    updateIterations_++;
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
            std::cout << "CLOSEST POINT " << closestPoints.closestPointWall1 << std::endl;
            for (uint16_t i = 0; i < pointCloud2D_.size(); ++i)
            {
                if (pointCloud2D_[i].x() > closestPoints.closestPointWall1.x() - wallWidth_ && pointCloud2D_[i].x() < closestPoints.closestPointWall1.x() + wallWidth_)
                {
                    pointCloud2D_.erase(pointCloud2D_.begin() + i);
                }
            }
            break;
    }
}

void GateDetection::clustering(std::vector<PointGroup>& clustersWall1, std::vector<PointGroup>& clustersWall2)
{
    // filter duplicates
    std::vector<Eigen::Vector2f> filteredPoints = filterDuplicatePoints(pointCloud2D_, 0.0001);

    // get closest point to bird
    this->closestPoints.closestPointWall1 = Maths::closestPoint(filteredPoints);
    
    if (filteredPoints.size() == 0)
    {
        std::cout << "FILTERED POINTS SIZE = 0" << std::endl;
        return;
    }

    // convert from Eigen to OpenCV
    std::vector<cv::Point2f> pointCloudCVWall1;
    std::vector<cv::Point2f> pointCloudCVWall2;
    for (auto point : filteredPoints)
    {
        if (point.x() >= closestPoints.closestPointWall1.x() && point.x() < closestPoints.closestPointWall1.x() + wallWidth_)
        {
            pointCloudCVWall1.push_back(Conversions::convertEigenToCVPoint<float>(point));
        } else
        {
            pointCloudCVWall2.push_back(Conversions::convertEigenToCVPoint<float>(point));
        }
    }

    // get closes point of the second wall
    this->closestPoints.closestPointWall2 = Maths::closestPoint(pointCloudCVWall2);

    // Check if we have more data poitns than clusters
    if (pointCloudCVWall1.size() < clustersWall1.size())
    {
        std::cout << "NOT ENOUGH DATA POINTS NEAR CLOSEST POINT" << std::endl;
        return;
    }

    bool computeGatePointForWall2 = true;
    if (pointCloudCVWall2.size() < clustersWall2.size())
    {
        std::cout << "WALL 2: NOT ENOUGH DATA POINTS NEAR CLOSEST POINT" << std::endl;
        computeGatePointForWall2 = false;
    }

    // WALL 1
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

    // WALL 2
    if (computeGatePointForWall2)
    {
        cv::Mat pointsMatWall2(pointCloudCVWall2.size(), 1, CV_32FC2, &pointCloudCVWall2[0]);

        // Perform K-Means clustering
        cv::Mat labelsWall2, centersWall2;
        cv::kmeans(pointsMatWall2, clustersWall2.size(), labelsWall2,
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
                10, cv::KMEANS_PP_CENTERS, centersWall2);

        // Separate points into clusters
        for (int i = 0; i < pointCloudCVWall2.size(); ++i) {
            int clusterIdx = labelsWall2.at<int>(i);
            clustersWall2[clusterIdx].points.push_back(pointCloudCVWall2[i]);
        }

        // compute centroids
        for (auto& cluster : clustersWall2)
        {
            cluster.centroid = Conversions::convertEigenToCVPoint<float>(Maths::mean(cluster.points));
        }

        // order clusters according to y-centroid from top to bottom
        std::sort(clustersWall2.begin(), clustersWall2.end(), [](const PointGroup& a, const PointGroup& b) -> bool {
            return a.centroid.y < b.centroid.y;
        });
    }
}

void GateDetection::clustering(std::vector<PointGroup>& clusters)
{
    // filter duplicates
    std::vector<Eigen::Vector2f> filteredPoints = filterDuplicatePoints(pointCloud2D_, 0.0001);

    // get closest point to bird
    this->closestPoints.closestPointWall1 = Maths::closestPoint(filteredPoints);
    
    if (filteredPoints.size() == 0)
    {
        std::cout << "FILTERED POINTS SIZE = 0" << std::endl;
        return;
    }

    // convert from Eigen to OpenCV
    std::vector<cv::Point2f> pointCloudCV;
    for (auto point : filteredPoints)
    {
        if (point.x() >= closestPoints.closestPointWall1.x() && point.x() < closestPoints.closestPointWall1.x() + wallWidth_)
        {
            pointCloudCV.push_back(Conversions::convertEigenToCVPoint<float>(point));
        }
    }

    if (pointCloudCV.size() < clusters.size())
    {
        std::cout << "NOT ENOUGH DATA POINTS NEAR CLOSEST POINT" << std::endl;
        return;
    }

    // convert points to Mat
    cv::Mat pointsMat(pointCloudCV.size(), 1, CV_32FC2, &pointCloudCV[0]);

    // Perform K-Means clustering
    cv::Mat labels, centers;
    cv::kmeans(pointsMat, clusters.size(), labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               10, cv::KMEANS_PP_CENTERS, centers);

    // Separate points into clusters
    for (int i = 0; i < pointCloudCV.size(); ++i) {
        int clusterIdx = labels.at<int>(i);
        clusters[clusterIdx].points.push_back(pointCloudCV[i]);
    }

    // compute centroids
    for (auto& cluster : clusters)
    {
        cluster.centroid = Conversions::convertEigenToCVPoint<float>(Maths::mean(cluster.points));
    }

    // order clusters according to y-centroid from top to bottom
    std::sort(clusters.begin(), clusters.end(), [](const PointGroup& a, const PointGroup& b) -> bool {
        return a.centroid.y < b.centroid.y;
    });
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
            if ((uniquePoint - point).norm() < threshold)
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

void GateDetection::renderMap()
{
    cv::Mat image = cv::Mat::zeros(mapWidth_, mapHeight_, CV_8UC3);

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

    // gate position
    cv::Point2f cvPointGate2 = Conversions::convertEigenToCVPoint<float>(gate2->position);
    cvPointGate2 = cv::Point2f(cvPointGate2.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointGate2.y);
    cv::circle(image, cvPointGate2 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(0, 0, 255), cv::FILLED);

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

void GateDetection::renderMap(const std::vector<PointGroup>& clusters, const std::vector<PointGroup>& hulls)
{
    cv::Mat image = cv::Mat::zeros(mapWidth_, mapHeight_, CV_8UC3);
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

    // gate position
    cv::Point2f cvPointGate2 = Conversions::convertEigenToCVPoint<float>(gate2->position);
    cvPointGate2 = cv::Point2f(cvPointGate2.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointGate2.y);
    cv::circle(image, cvPointGate2 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(0, 0, 255), cv::FILLED);

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

    // gate position
    cv::Point2f cvPointGate2 = Conversions::convertEigenToCVPoint<float>(gate2->position);
    cvPointGate2 = cv::Point2f(cvPointGate2.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointGate2.y);
    cv::circle(image, cvPointGate2 * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(0, 0, 255), cv::FILLED);

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

void GateDetection::getGatePosition(const std::vector<PointGroup>& hullsWall1, const std::vector<PointGroup>& hullsWall2)
{
    float maxGapWall1 = 0;
    float maxGapWall2 = 0;
    for (size_t i = 0; i < hullsWall1.size() - 1; ++i)
    {
        float min1 = std::numeric_limits<float>::max();
        float max1 = std::numeric_limits<float>::lowest();
        float min2 = std::numeric_limits<float>::max();
        float max2 = std::numeric_limits<float>::lowest();

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

        min1 = std::numeric_limits<float>::max();
        max1 = std::numeric_limits<float>::lowest();
        min2 = std::numeric_limits<float>::max();
        max2 = std::numeric_limits<float>::lowest();

        for (const auto& point1 : hullsWall2[i].points)
        {
            if (point1.y < min1) min1 = point1.y;
            if (point1.y > max1) max1 = point1.y;
        }

        for (const auto& point2 : hullsWall2[i + 1].points)
        {
            if (point2.y < min2) min2 = point2.y;
            if (point2.y > max2) max2 = point2.y;
        }

        gap = std::max(min1, min2) - std::min(max1, max2);
        if (gap > maxGapWall2)
        {
            maxGapWall2 = gap;
            gate2->lowerBoundY = std::min(max1, max2);
            gate2->upperBoundY = std::max(min1, min2);
        }
    }
    
    if (std::abs(gate1->upperBoundY - gate1->lowerBoundY) > 0.3)
    {
        gate1->position = Eigen::Vector2f(closestPoints.closestPointWall1.x(), (gate1->upperBoundY + gate1->lowerBoundY)/2.0f);
    }

    if (std::abs(gate2->upperBoundY - gate2->lowerBoundY) > 0.3)
    {
        gate2->position = Eigen::Vector2f(closestPoints.closestPointWall2.x(), (gate2->upperBoundY + gate2->lowerBoundY)/2.0f);
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
}