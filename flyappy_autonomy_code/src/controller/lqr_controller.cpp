#include "flyappy_autonomy_code/controller/lqr_controller.hpp"

LQR::LQR()
{
    maxIterations_ = 1000;
    Q_ = Eigen::Matrix4f::Zero();
    R_ = Eigen::Matrix2f::Zero();
}

LQR::LQR(const Eigen::Matrix4f& Q, const Eigen::Matrix2f& R, int maxIterations)
{
    maxIterations_ = maxIterations;
    Q_ = Q;
    R_ = R;

    computeK();
}

LQR::LQR(const Eigen::Matrix4f& Q, const Eigen::Matrix2f& R, int maxIterations, const Eigen::Matrix<float, 2, 4>& K)
{
    maxIterations_ = maxIterations;
    Q_ = Q;
    R_ = R;
    K_ = K;
}

void LQR::computeK()
{
    Eigen::Matrix4f P = Q_;
    Eigen::Matrix4f P_next;
    double tolerance = 1e-12;

    // Solve Riccati Difference Equation
    for (unsigned int i = 0; i < maxIterations_; ++i)
    {
        Eigen::Matrix<float, 2, 4> K_temp = (system_.Bd.transpose() * P * system_.Bd + R_).inverse() * system_.Bd.transpose() * P * system_.Ad;
        P_next = system_.Ad.transpose() * P * system_.Ad + Q_ - system_.Ad.transpose() * P * system_.Bd * K_temp;

        if ((P_next - P).cwiseAbs().maxCoeff() < tolerance) 
        {
            P = P_next;
            break;
        }

        P = P_next;
    }
    
    // Compute optimal control K_
    K_ = (system_.Bd.transpose() * P * system_.Bd + R_).inverse() * system_.Bd.transpose() * P * system_.Ad;
}

Eigen::Vector2f LQR::eval(const Eigen::Vector4f& deltaXk)
{
    return -K_ * deltaXk;
}