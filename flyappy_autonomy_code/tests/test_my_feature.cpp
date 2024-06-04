#include <gtest/gtest.h>

#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "flyappy_autonomy_code/agent/flyappy.hpp"
#include "flyappy_autonomy_code/perception/gate_detection.hpp"
#include "flyappy_autonomy_code/utils/conversion.hpp"

#include "flyappy_autonomy_code/controller/lqr_controller.hpp"
#include "flyappy_autonomy_code/controller/mpc_controller.hpp"

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

TEST(ControllerTesting, qpoasesTests)
{
    // Example QPOASES to test if library is properly installed
    USING_NAMESPACE_QPOASES

	/* Setup data of first QP. */
	real_t H[2*2] = { 1.0, 0.0, 0.0, 0.5 };
	real_t A[1*2] = { 1.0, 1.0 };
	real_t g[2] = { 1.5, 1.0 };
	real_t lb[2] = { 0.5, -2.0 };
	real_t ub[2] = { 5.0, 2.0 };
	real_t lbA[1] = { -1.0 };
	real_t ubA[1] = { 2.0 };

	/* Setup data of second QP. */
	real_t g_new[2] = { 1.0, 1.5 };
	real_t lb_new[2] = { 0.0, -1.0 };
	real_t ub_new[2] = { 5.0, -0.5 };
	real_t lbA_new[1] = { -2.0 };
	real_t ubA_new[1] = { 1.0 };


	/* Setting up QProblem object. */
	QProblem example( 2,1 );

	Options options;
	example.setOptions( options );

	/* Solve first QP. */
	int_t nWSR = 10;
	example.init( H,g,A,nullptr,nullptr,lbA,ubA, nWSR );

	/* Get and print solution of first QP. */
	real_t xOpt[2];
	real_t yOpt[2+1];
	example.getPrimalSolution( xOpt );
	example.getDualSolution( yOpt );
	printf( "\nxOpt = [ %e, %e ];  yOpt = [ %e, %e, %e ];  objVal = %e\n\n", 
			xOpt[0],xOpt[1],yOpt[0],yOpt[1],yOpt[2],example.getObjVal() );
	
	/* Solve second QP. */
	nWSR = 10;
	example.hotstart( g_new,lb_new,ub_new,lbA_new,ubA_new, nWSR );

	/* Get and print solution of second QP. */
	example.getPrimalSolution( xOpt );
	example.getDualSolution( yOpt );
	printf( "\nxOpt = [ %e, %e ];  yOpt = [ %e, %e, %e ];  objVal = %e\n\n", 
			xOpt[0],xOpt[1],yOpt[0],yOpt[1],yOpt[2],example.getObjVal() );

	example.printOptions();
	/*example.printProperties();*/

	/*getGlobalMessageHandler()->listAllMessages();*/

}

TEST(ControllerTesting, convertEigenToReal)
{
    Eigen::Matrix<float, 4, 2> test;
    test << 1, 2,
            5, 6,
            9, 10,
            13, 14;
    
    real_t* T = Conversions::convertEigenToRealT<float>(test);

    // print
    for (int i = 0; i < 8; ++i)
    {
        std::cout << T[i] << std::endl;
    }

    delete[] T;
}

TEST(ControllerTesting, matrixFormation)
{
    using namespace Eigen;
    USING_NAMESPACE_QPOASES
    // Matrix2d block1;
    // block1 << 1, 2,
    //           3, 4;

    // Matrix3d block2;
    // block2 << 5, 6, 7,
    //           8, 9, 10,
    //           11, 12, 13;

    // // Calculate the size of the block diagonal matrix
    // int totalRows = block1.rows() + block2.rows();
    // int totalCols = block1.cols() + block2.cols();

    // // Initialize the block diagonal matrix
    // MatrixXd blockDiagonalMatrix = MatrixXd::Zero(totalRows, totalCols);

    // // Place the blocks in the block diagonal matrix
    // blockDiagonalMatrix.block(0, 0, block1.rows(), block1.cols()) = block1;
    // blockDiagonalMatrix.block(block1.rows(), block1.cols(), block2.rows(), block2.cols()) = block2;

    // // Print the block diagonal matrix
    // std::cout << "Block Diagonal Matrix:\n" << blockDiagonalMatrix << std::endl;


    /// setting up Q = blockdiag(Q, Q, ..., Q, P);
    Eigen::Matrix4d Q;
    Q << 1000, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1000, 0,
         0, 0, 0, 1;
    Eigen::Matrix2d R;
    R << 1, 0,
         0, 0.001;

    MPC mpc(4, 2, 5, 10);
    mpc.setQRPMatrices(Q, R);
    // mpc.constructQBar();

    std::cout << "Q_bar MPC: \n" << mpc.QBar_ << std::endl;

    // mpc.constructRBar();
    std::cout << "R_bar MPC: \n" << mpc.RBar_ << std::endl;

    // Construct Sx and Su
    // mpc.constructSxSu();

    std::cout << "Sx MPC: \n" << mpc.Sx_ << std::endl;
    std::cout << "Su MPC: \n" << mpc.Su_ << std::endl;

    // UP TO HERE I double checked with matlab; Since the one below is based on calculation I assume that is correct.
    // construct H, Y, F
    // mpc.constructF();
    std::cout << "F: \n" << mpc.F_ << std::endl;

    // mpc.constructH();
    std::cout << "H: \n" << mpc.H_ << std::endl;

    // mpc.constructY();
    std::cout << "Y: \n" << mpc.Y_ << std::endl;

    // Constraint matrix eigenA_
    // mpc.constructConstraintsMatrix();
    std::cout << "Eigen A: \n" << mpc.eigenA_ << std::endl;

    // upperbouand 
    // mpc.constructUpperBoundConstraints();
}

TEST(ControllerTesting, MPCTest)
{
    Eigen::Matrix4d Q;
    Q << 1000, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1000, 0,
         0, 0, 0, 1;
    Eigen::Matrix2d R;
    R << 1, 0,
         0, 0.001;

    MPC mpc(4, 2, 30, 100);
    mpc.setQRPMatrices(Q, R);

    // solve mpc
    Eigen::Vector4f Xk(0, 0, 0, 0);
    Eigen::Vector4f XRef(0, 0, 0.5, 0);
    Eigen::Vector2d U;

    for (int i = 0; i < 4; ++i)
    {
        Eigen::MatrixXd g = mpc.F_.transpose() * Xk.cast<double>();
        mpc.g_ = Conversions::convertEigenToRealT(g);
        // mpc.setQRPMatrices(Q, R);
        mpc.constructH();
        mpc.constructUpperBoundConstraints();
        mpc.constructConstraintsMatrix();
        QProblem qp(mpc.Nu_ * mpc.N_, 4 * mpc.N_, HST_POSDEF);
        Options options;
        options.setToMPC();
        options.printLevel = PL_HIGH;
        qp.setOptions(options);

        int nWSR = mpc.nWSR_;
        std::cout << mpc.nWSR_ << std::endl;
        qp.init(mpc.H_, mpc.g_, mpc.A_, nullptr, nullptr, mpc.lbA_, mpc.ubA_, nWSR);
        real_t* xOpt = new real_t[mpc.Nu_ * mpc.N_];
        qp.getPrimalSolution(xOpt);
        U(0) = xOpt[0];
        U(1) = xOpt[1];
        delete[] xOpt;
        delete[] mpc.g_;
        delete[] mpc.H_;
        delete[] mpc.A_;
        delete[] mpc.lbA_;
        delete[] mpc.ubA_;
        mpc.g_ = nullptr;
        mpc.H_ = nullptr;
        mpc.A_ = nullptr;
        mpc.lbA_ = nullptr;
        mpc.ubA_ = nullptr;
        std::cout << "U: " << U << std::endl;
    }

    for (int i = 0; i < 5; ++i)
    {
        mpc.solve(Xk.cast<double>(), U);
        std::cout << "U: " << U << std::endl;
    }

}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
