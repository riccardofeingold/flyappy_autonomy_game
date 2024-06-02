#include "flyappy_autonomy_code/agent/flyappy.hpp"

Flyappy::Flyappy() : 
    pid_(PIDController()),
    gateDetector_(GateDetection()),
    mpc_(MPCController())
{
    stateEstimator_ = std::make_shared<StateEstimation>(SAMPLING_TIME);
    velMeasured_ = Eigen::Vector2f::Zero();

    // init variables
    XRef_ = Eigen::Vector4d::Zero();

    // initialize the LQR Controller
    // Eigen::Matrix4f QLQR;
    // QLQR << 100, 0, 0, 0,
    //      0, 1, 0, 0,
    //      0, 0, 300, 0,
    //      0, 0, 0, 1;
    // Eigen::Matrix2f RLQR;
    // RLQR << 50, 0,
    //      0, 1;

    // lqr_ = LQR(QLQR, RLQR, 1000);

    // initialize MPC Controller
    Eigen::Matrix4d QMPC;
    QMPC << 1000, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1000, 0,
         0, 0, 0, 1;
    Eigen::Matrix2d RMPC;
    RMPC << 1, 0,
         0, 0.001;
    mpc_.setQRPMatrices(QMPC, RMPC);
}

void Flyappy::init()
{
    if (laserData_.intensities.empty())
    {
        return;
    }

    if (laserData_.intensities[laserData_.ranges.size() - 1] == 0)
    {
        XRef_ = Eigen::Vector4d(0, 0, 0.5, 0);
    } 
    else
    {
        gateDetector_.computeBoundaries();
        stateChanged = true;
        currentState_ = States::MOVE_FORWARD;
    }
}

void Flyappy::update()
{
    stateEstimator_->update(velMeasured_);
    gateDetector_.update(stateEstimator_->getPosition(), laserData_, currentState_);

    if (currentState_ == States::INIT)
    {
        std::cout << "STATE: INIT" << std::endl;
        init();
    } else if (currentState_ == States::MOVE_FORWARD)
    {
        if (stateChanged)
        {
            std::cout << "STATE: MOVE_FORWARD" << std::endl;
            gateDetector_.reset(ResetStates::CLEAR_ALL);
            stateChanged = false;
        }

        XRef_ = Eigen::Vector4d(4.7, 0, 0, 0);
        
        if (stateEstimator_->getVelocity().norm() < 0.005 && stateEstimator_->getPosition().x() > XRef_(0))
        {
            stateChanged = true;
            currentState_ = States::EXPLORE;
        }
    } 
    else if (currentState_ == States::EXPLORE)
    {
        if (stateChanged)
        {
            std::cout << "STATE: EXPLORE" << std::endl;
            gateDetector_.update(stateEstimator_->getPosition(), laserData_, currentState_);
            stateChanged = false;
        }

        // Switch to TARGET mode if we reach already safety distance
        double xref = gateDetector_.closestPoints.closestPointWall1.x() - pipeGap;
        if (stateEstimator_->getPosition().x() > xref)
        {
            stateChanged = true;
            currentState_ = States::TARGET;
            return;
        }

        XRef_ = Eigen::Vector4d(xref, 2.0, stateEstimator_->getPosition().y(), 0.0);
        if (std::abs(stateEstimator_->getPosition().y() - gateDetector_.gate1->position.y()) > 0.6) XRef_(1) = -3.0;

    } else if (currentState_ == States::TARGET)
    {
        if (stateChanged)
        {
            std::cout << "STATE: TARGET" << std::endl;
            stateChanged = false;
        }

        // "Path planning"
        if (std::abs(stateEstimator_->getPosition().y() - gateDetector_.gate1->position.y()) < 0.01)
        {
            if (stateEstimator_->getPosition().x() > gateDetector_.closestPoints.closestPointWall1.x() + wallWidth)
            {
                stateChanged = true;
                currentState_ = States::EXPLORE;
                gateDetector_.update(stateEstimator_->getPosition(), laserData_, currentState_);
                return;
            }

            XRef_ = Eigen::Vector4d(gateDetector_.gate1->position.x() + wallWidth, 3.0, stateEstimator_->getPosition().y(), 0);
        } else
        {
            XRef_ = Eigen::Vector4d(gateDetector_.gate1->position.x() - pipeGap, 3.0, gateDetector_.gate1->position.y(), 0);
        }

        // std::cout << "XREF: " << XRef_ << std::endl;
    }
    
    // Track Path: Compute control input
    controlInput_ = Eigen::Vector2d::Zero();
    bool success = mpc_.solve(stateEstimator_->getStateVector().cast<double>() - XRef_, controlInput_);
    if (!success) std::cout << "ERROR: NO U optimal found" << std::endl;
}

Eigen::Vector2d Flyappy::getControlInput()
{
    return controlInput_;
}