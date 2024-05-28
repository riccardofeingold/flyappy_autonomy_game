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
    gate = std::make_unique<Gate>();
}

void GateDetection::update(const Eigen::Vector2f& position, const sensor_msgs::LaserScan& laserData)
{
    // reset based on travelled distance
    if (pointCloud2D_.size() > pointcloudBufferSize_)
    {
        std::cout << "CLEAR OLDEST" << std::endl;
        reset(ResetStates::CLEAR_OLDEST);
    } else if (position.x() > gate->position.x() + wallWidth_/2)
    {
        std::cout << "CLEAR ALL NEAR CLOSEST POINT" << std::endl;
        
        gate->prevPosition = gate->position;
        reset(ResetStates::CLEAR_ALL_NEAR_CLOSEST_POINT);

        std::vector<Eigen::Vector2f> filteredPoints = filterDuplicatePoints(pointCloud2D_, 0.0001);
        this->closestPoint = Maths::closestPoint(filteredPoints);
        gate->position[0] = this->closestPoint.x();
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
        std::vector<PointGroup> clusters(numClusters_);
        clustering(clusters);

        bool enoughDataPoints;
        std::vector<PointGroup> hulls(numClusters_);
        convexHull(clusters, hulls, enoughDataPoints);
        
        if (enoughDataPoints)
        {
            renderMap(clusters, hulls);
            getGatePosition(hulls);
        }
        else
        {
            std::cout << "NOT ENOUGH DATA POINTS FOR CONVEX HULL" << std::endl;
            renderMap(clusters);
            getGatePosition(clusters);
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
            std::cout << "CLOSEST POINT " << closestPoint << std::endl;
            for (uint16_t i = 0; i < pointCloud2D_.size(); ++i)
            {
                if (pointCloud2D_[i].x() > closestPoint.x() - wallWidth_ && pointCloud2D_[i].x() < closestPoint.x() + wallWidth_)
                {
                    pointCloud2D_.erase(pointCloud2D_.begin() + i);
                }
            }
            break;
    }
}

void GateDetection::clustering(std::vector<PointGroup>& clusters)
{
    // filter duplicates
    std::vector<Eigen::Vector2f> filteredPoints = filterDuplicatePoints(pointCloud2D_, 0.0001);

    // get closest point to bird
    this->closestPoint = Maths::closestPoint(filteredPoints);
    
    if (filteredPoints.size() == 0)
    {
        std::cout << "FILTERED POINTS SIZE = 0" << std::endl;
        return;
    }

    // convert from Eigen to OpenCV
    std::vector<cv::Point2f> pointCloudCV;
    for (auto point : filteredPoints)
    {
        if (point.x() >= closestPoint.x() && point.x() < closestPoint.x() + wallWidth_)
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
    cv::Point2f cvPointGate = Conversions::convertEigenToCVPoint<float>(gate->position);
    cvPointGate = cv::Point2f(cvPointGate.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointGate.y);
    cv::circle(image, cvPointGate * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(0, 0, 255), cv::FILLED);

    // closest point
    cv::Point2f cvPointClosest = Conversions::convertEigenToCVPoint<float>(closestPoint);
    cvPointClosest = cv::Point2f(cvPointClosest.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointClosest.y);
    cv::circle(image, cvPointClosest * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(255, 0, 0), cv::FILLED);

    cv::imshow("Point Cloud", image);
    cv::waitKey(1);
}

void GateDetection::renderMap(const std::vector<PointGroup>& clusters, const std::vector<PointGroup>& hulls)
{
    cv::Mat image = cv::Mat::zeros(mapWidth_, mapHeight_, CV_8UC3);
    cv::Scalar color = cv::Scalar(0, 255, 0);

    for (uint8_t i = 0; i < clusters.size(); ++i)
    {
        for (uint16_t ii = 0; ii < hulls[i].points.size(); ++ii)
        {
            cv::Point2f startPoint = cv::Point2f(hulls[i].points[ii].x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -hulls[i].points[ii].y);
            cv::Point2f endPoint = cv::Point2f(hulls[i].points[(ii + 1) % hulls[i].points.size()].x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -hulls[i].points[(ii + 1) % hulls[i].points.size()].y);
            cv::line(image, startPoint * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), endPoint * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), color, 2);
        }

        for (const auto& point : clusters[i].points)
        {
            cv::Point2f p(point.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -point.y);
            cv::circle(image, p * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, color, cv::FILLED);
        }
    }
    
    // gate position
    cv::Point2f cvPointGate = Conversions::convertEigenToCVPoint<float>(gate->position);
    cvPointGate = cv::Point2f(cvPointGate.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointGate.y);
    cv::circle(image, cvPointGate * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(0, 0, 255), cv::FILLED);

    // closest point
    cv::Point2f cvPointClosest = Conversions::convertEigenToCVPoint<float>(closestPoint);
    cvPointClosest = cv::Point2f(cvPointClosest.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointClosest.y);
    cv::circle(image, cvPointClosest * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(255, 0, 0), cv::FILLED);

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
    cv::Point2f cvPointGate = Conversions::convertEigenToCVPoint<float>(gate->position);
    cvPointGate = cv::Point2f(cvPointGate.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointGate.y);
    cv::circle(image, cvPointGate * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(0, 0, 255), cv::FILLED);

    // closest point
    cv::Point2f cvPointClosest = Conversions::convertEigenToCVPoint<float>(closestPoint);
    cvPointClosest = cv::Point2f(cvPointClosest.x - pixelInMeters * countPointcloudWindowUpdates_ * mapWidth_, -cvPointClosest.y);
    cv::circle(image, cvPointClosest * (int) 1 / pixelInMeters + cv::Point2f(0, mapHeight_/2), 3, cv::Scalar(255, 0, 0), cv::FILLED);

    cv::imshow("Point Cloud", image);
    cv::waitKey(1);
}

void GateDetection::getGatePosition(const std::vector<PointGroup>& hulls)
{
    float maxGap = 0;
    for (size_t i = 0; i < hulls.size() - 1; ++i)
    {
        const auto& hull1 = hulls[i];
        const auto& hull2 = hulls[i + 1];

        float min1 = std::numeric_limits<float>::max();
        float max1 = std::numeric_limits<float>::lowest();
        float min2 = std::numeric_limits<float>::max();
        float max2 = std::numeric_limits<float>::lowest();

        for (const auto& point1 : hull1.points)
        {
            if (point1.y < min1) min1 = point1.y;
            if (point1.y > max1) max1 = point1.y;
        }

        for (const auto& point2 : hull2.points)
        {
            if (point2.y < min2) min2 = point2.y;
            if (point2.y > max2) max2 = point2.y;
        }

        float gap = std::max(min1, min2) - std::min(max1, max2);
        if (gap > maxGap)
        {
            maxGap = gap;
            gate->lowerBoundY = std::min(max1, max2);
            gate->upperBoundY = std::max(min1, min2);
        }
    }
    
    if (std::abs(gate->upperBoundY - gate->lowerBoundY) > 0.3)
    {
        gate->position = Eigen::Vector2f(closestPoint.x(), (gate->upperBoundY + gate->lowerBoundY)/2.0f);
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