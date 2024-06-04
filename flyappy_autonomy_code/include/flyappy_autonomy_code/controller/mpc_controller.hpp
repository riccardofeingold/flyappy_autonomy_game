#pragma once

#include <qpOASES.hpp>
#include <gtest/gtest.h>

// Custom
#include "flyappy_autonomy_code/agent/system_dynamics.hpp"
#include "flyappy_autonomy_code/utils/conversion.hpp"

USING_NAMESPACE_QPOASES
class MPC 
{
    public:
    MPC(
        int Nx=4, // state dimension
        int Nu=2, // control dimension
        int N=30, // horizon: If you use only Input constraint set it to 30 otherwise 70
        int nWSR=1500, // number of working set calculations
        bool onlyInputConstraints = false
    );
    
    ~MPC();

    /// @brief set State matrix constraints
    void setStateMatrixConstraints(const Eigen::MatrixXd& Hx, const Eigen::VectorXd& hx);

    /// @brief set State matrix constraints
    void setTerminalMatrixConstraints(const Eigen::MatrixXd& Hf, const Eigen::VectorXd& hf);
    
    /// @brief compute steady state for specific reference
    Eigen::VectorXd computeSteadyState(const Eigen::Vector4d& r);

    /// @brief set Q, P, and R matrix
    void setQRPMatrices(const Eigen::Matrix4d& Q, const Eigen::Matrix2d& R);

    /// @brief solve the QP problem and return optimal U
    bool solve(const Eigen::Vector4d& Xk, const Eigen::Vector4d& Xs, const Eigen::Vector2d& Us, Eigen::Vector2d& U);

    private:
    /// @brief Construct the constraints matrix
    void constructConstraintsMatrix();

    /// @brief Construct the upperbound constraints
    void constructUpperBoundConstraints(const Eigen::Vector2d& Us, const Eigen::Vector4d& Xs);

    /// @brief construct blockdiagonal matrix Q bar = blockdiag(Q, ..., Q, P)
    void constructQBar();

    /// @brief construct blockdiagonal matrix R bar = blockdiag(R, ..., R)
    void constructRBar();

    /// @brief construct Sx and Su
    void constructSxSu();

    /// @brief construct H = Su^T * Q_bar * Su + R_bar
    void constructH();

    /// @brief construct F = Sx^T * Q_bar * Su
    void constructF();

    /// @brief construct Y = Sx^T * Q_bar * Sx
    void constructY();

    /// @brief construct E = [0 ... 0 -Hx -Hx*A ... -Hf*A^N]
    void constructE();
    
    /// @brief compute P
    void computeP();

    // Settings
    const int N_;
    const int Nx_;
    const int Nu_;
    bool onlyInputConstraints_;
    int nWSR_;
    Eigen::Matrix4d Q_;
    Eigen::MatrixXd QBar_;
    Eigen::Matrix2d R_;
    Eigen::MatrixXd RBar_;
    Eigen::Matrix4d P_;

    Eigen::MatrixXd eigenH_;
    real_t* H_ = nullptr;
    real_t* g_ = nullptr;
    Eigen::MatrixXd eigenA_;
    real_t* A_ = nullptr;
    real_t* lbA_ = nullptr;
    Eigen::VectorXd eigenUBA_;
    real_t* ubA_ = nullptr;

    // State matrices that occur when substituting the dynamics: x(k) = S_x * x(0) + S_u * U
    Eigen::MatrixXd Sx_;
    Eigen::MatrixXd Su_;

    // Variables to simplify equations
    Eigen::MatrixXd F_;
    Eigen::MatrixXd Y_;
    Eigen::MatrixXd E_;

    // Constraints
    /// @brief State Matrix constraints
    Eigen::MatrixXd Hx_;
    /// @brief RHS State matrix constraints
    Eigen::VectorXd hx_;
    /// @brief Input Constraint Matrix
    Eigen::Matrix<double, 4, 2> Hu_; // Fix
    /// @brief RHS of input constraint inequality
    Eigen::Vector4d hu_; // Fix
    /// @brief terminal constraint
    Eigen::MatrixXd Hf_;
    /// @brief terminal constraint RHS
    Eigen::VectorXd hf_;

    /// @brief Accessing System Dynamics
    SystemDynamics system_;

    // TESTING
    FRIEND_TEST(ControllerTesting, matrixFormation);
    FRIEND_TEST(ControllerTesting, MPCTest);
};