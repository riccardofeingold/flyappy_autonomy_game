#include "flyappy_autonomy_code/agent/flyappy.hpp"

Flyappy::Flyappy() : 
    pid_(PIDController()),
    gateDetector_(GateDetection()),
    mpc_(MPCController())
{
    stateEstimator_ = std::make_shared<StateEstimation>(SAMPLING_TIME);
    velMeasured_ = Eigen::Vector2f::Zero();
    
    // initialize the LQR Controller
    Eigen::Matrix4f QLQR;
    QLQR << 100, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 300, 0,
         0, 0, 0, 1;
    Eigen::Matrix2f RLQR;
    RLQR << 50, 0,
         0, 1;

    lqr_ = LQR(QLQR, RLQR, 1000);

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
    stateEstimator_->update(velMeasured_);
    gateDetector_.update(stateEstimator_->getPosition(), laserData_);

    if (laserData_.intensities.empty())
    {
        return;
    }

    if (laserData_.intensities[laserData_.ranges.size() - 1] == 0 && !gateDetector_.initDone)
    {
        Eigen::Vector4f XRef(0, 0, 0.5, 0);
        bool success = mpc_.solve((stateEstimator_->getStateVector() - XRef).cast<double>(), controlInput_);
        if (!success) std::cout << "ERROR: NO U optimal found" << std::endl;
        // controlInput_ = lqr_.eval(stateEstimator_->getStateVector() - XRef).cast<double>();
        // controlInput_ = pid_.computeAcceleration(Eigen::Vector2f(0, 0.5), Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());
    } 
    else
    {
        gateDetector_.computeBoundaries();
        gateDetector_.initDone = true;
        // bool success = mpc_.solve(stateEstimator_->getStateVector().cast<double>(), controlInput_);
        controlInput_ = lqr_.eval(stateEstimator_->getStateVector()).cast<double>();
        // controlInput_ = pid_.computeAcceleration(Eigen::Vector2f::Zero(), Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());

        if (stateEstimator_->getVelocity().norm() < 0.005)
        {
            stateChanged = true;
            currentState_ = States::MOVE_FORWARD;
        }
    }
}

void Flyappy::update()
{
    if (currentState_ == States::INIT)
    {
        std::cout << "STATE: INIT" << std::endl;
        init();
    } else if (currentState_ == States::MOVE_FORWARD)
    {
        if (stateChanged)
        {
            std::cout << "STATE: MOVE_FORWARD" << std::endl;
            // DO SOMETHING BEFORE MOVE_FORWARD MODE STARTS
            gateDetector_.reset(ResetStates::CLEAR_ALL);
            stateChanged = false;
        }

        stateEstimator_->update(velMeasured_);
        gateDetector_.update(stateEstimator_->getPosition(), laserData_);
        
        Eigen::Vector4d XRef(4.5, 0, 0, 0);
        bool success = mpc_.solve(stateEstimator_->getStateVector().cast<double>() - XRef, controlInput_);
        // Eigen::Vector2f refPos(3.5, 0);
        // controlInput_ = pid_.computeAcceleration(refPos, Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());

        if (stateEstimator_->getVelocity().norm() < 0.005 && stateEstimator_->getPosition().x() > XRef(0))
        {
            stateChanged = true;
            currentState_ = States::FLY;
        }
    } 
    else if (currentState_ == States::FLY)
    {
        if (stateChanged)
        {
            // DO SOMETHING BEFORE FLY MODE STARTS
            std::cout << "STATE: FLY" << std::endl;
            stateChanged = false;
        }
        // Updating the current state: Position
        stateEstimator_->update(velMeasured_);
        
        // Updating map and checking for Gate
        gateDetector_.update(stateEstimator_->getPosition(), laserData_);

        // Track Path:
        controlInput_ = Eigen::Vector2d::Zero();
        Eigen::Vector4f XRef = Eigen::Vector4f(gateDetector_.gate2->position.x() - 0.5, 0, gateDetector_.gate1->position.y(), 0);
        bool success = mpc_.solve((stateEstimator_->getStateVector() - XRef).cast<double>(), controlInput_);
        // controlInput_ = lqr_.eval(stateEstimator_->getStateVector() - XRef);
    }
}

Eigen::Vector2d Flyappy::getControlInput()
{
    return controlInput_;
}