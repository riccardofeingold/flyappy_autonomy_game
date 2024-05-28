#include <gtest/gtest.h>

#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "flyappy_autonomy_code/agent/flyappy.hpp"
#include "flyappy_autonomy_code/perception/gate_detection.hpp"
#include "flyappy_autonomy_code/utils/conversion.hpp"

#include "flyappy_autonomy_code/controller/lqr_controller.hpp"

TEST(Conversion, CVPointEigen)
{
    cv::Point2f cvPoint1(10, 20);
    cv::Point2d cvPoint2(-1, 3);
    Eigen::Vector2f eigenPoint3(3, 6);
    Eigen::Vector2d eigenPoint4(7, 4);

    Eigen::Vector2f test1 = Conversions::convertCVPointToEigen<float>(cvPoint1);
    Eigen::Vector2d test4 = Conversions::convertCVPointToEigen<double>(cvPoint2);

    cv::Point2f test5 = Conversions::convertEigenToCVPoint<float>(eigenPoint3);
    cv::Point2d test8 = Conversions::convertEigenToCVPoint<double>(eigenPoint4);
}

TEST(OpenCV, ClusteringTest)
{
    // Generate random points
    cv::RNG rng(0);
    std::vector<cv::Point2f> points;
    for (int i = 0; i < 30; ++i) {
        points.emplace_back(rng.uniform(-5.0f, 5.0f) + 5, rng.uniform(-5.0f, 5.0f) + 5);
        points.emplace_back(rng.uniform(-5.0f, 5.0f) - 5, rng.uniform(-5.0f, 5.0f) - 5);
        points.emplace_back(rng.uniform(-5.0f, 5.0f) + 5, rng.uniform(-5.0f, 5.0f) - 5);
    }

    // Convert points to Mat
    cv::Mat pointsMat(points.size(), 1, CV_32FC2, &points[0]);

    // Perform K-Means clustering
    int numClusters = 2;
    cv::Mat labels, centers;
    cv::kmeans(pointsMat, numClusters, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);

    // Separate points into clusters and compute convex hull for each cluster
    std::vector<std::vector<cv::Point2f>> clusters(numClusters);
    for (int i = 0; i < points.size(); ++i) {
        int clusterIdx = labels.at<int>(i);
        clusters[clusterIdx].push_back(points[i]);
    }

    cv::Mat image = cv::Mat::zeros(500, 500, CV_8UC3);
    std::vector<cv::Scalar> colors = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255)};

    for (int i = 0; i < numClusters; ++i) {
        std::vector<cv::Point2f> hull;
        ASSERT_GE(clusters[i].size(), 3);
        if (clusters[i].size() >= 3) { // Need at least 3 points to form a convex hull
            cv::convexHull(clusters[i], hull);
            for (int j = 0; j < hull.size(); ++j) {
                cv::line(image, hull[j] * 10 + cv::Point2f(250, 250), hull[(j + 1) % hull.size()] * 10 + cv::Point2f(250, 250), colors[i], 2);
            }
        }
        for (const auto& pt : clusters[i]) {
            cv::circle(image, pt * 10 + cv::Point2f(250, 250), 3, colors[i], cv::FILLED);
        }
    }

    // Display the result
    cv::imshow("Convex Hulls for Multiple Clusters", image);
    cv::waitKey(0);
}


TEST(ControllerTesting, LQRTest)
{
    Eigen::Matrix4f Q;
    Q << 100, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 100, 0,
         0, 0, 0, 1;
    std::cout << Q.maxCoeff() << std::endl;
    Eigen::Matrix2f R;
    R << 200, 0,
         0, 1;

    LQR lqr = LQR(Q, R, 1000);
    std::cout << "K: \n" << lqr.K_ << std::endl;

    Eigen::Vector4f X0(0, 0, 0, 0);
    Eigen::Vector4f XRef(2, 0, -2, 0);

    const int horizon = 100;
    Eigen::Matrix<float, 4, horizon + 1> X = Eigen::Matrix<float, 4, horizon + 1>::Zero();
    Eigen::Matrix<float, 2, horizon> U = Eigen::Matrix<float, 2, horizon>::Zero();

    X.col(0) = X0;
    Eigen::Vector4f Xk = X0;
    for (uint16_t i = 1; i < 100; ++i)
    {
        Eigen::Vector2f Uk = lqr.eval(X0);
        Xk = lqr.system_.nextState(Uk, Xk - XRef);
        X.col(i) = Xk;
        U.col(i-1) = Uk;
    }
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
