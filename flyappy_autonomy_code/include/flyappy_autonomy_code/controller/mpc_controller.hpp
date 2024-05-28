#pragma once

#include <qpOASES.hpp>

using namespace qpOASES;

class MPCController 
{
    public:
    MPCController(
        int Nx, // state dimension
        int Nu, // control dimension
        int N // horizon
    );
    
    ~MPCController();

    private:
    /// @brief Construct the H matrix based on the Q and R matrices used in the quadratice cost function
    /// @param Q
    /// @param R
    void setProblemData(const real_t* Q, const real_t* R);

    /// @brief 
};