#include "flyappy_autonomy_code/agent/system_dynamics.hpp"

SystemDynamics::SystemDynamics()
{
    A << 0, 1, 0, 0,
         0, 0, 0, 0,
         0, 0, 0, 1,
         0, 0, 0, 0;
    
    Nx = A.rows();

    Ad << 1, SAMPLING_TIME, 0, 0,
         0, 1, SAMPLING_TIME, 0,
         0, 0, 1, SAMPLING_TIME,
         0, 0, 0, 1;

    B << 0, 0,
         1, 0,
         0, 0,
         0, 1;
    
    Nu = B.cols();

    Bd << 0.5*SAMPLING_TIME*SAMPLING_TIME, 0,
          SAMPLING_TIME, 0,
          0, 0.5*SAMPLING_TIME*SAMPLING_TIME,
          0, SAMPLING_TIME;
    
    C << 1, 0, 0, 0,
         0, 0, 1, 0;

    Ny = C.rows();

    Cd = C;
}

Eigen::Vector4f SystemDynamics::nextState(const Eigen::Vector2f& Uk, const Eigen::Vector4f& Xk)
{
    return Ad * Xk + Bd * Uk;
}

Eigen::Vector2f SystemDynamics::getOnlyPosition(const Eigen::Vector4f& Xk)
{
    return Cd * Xk;
}