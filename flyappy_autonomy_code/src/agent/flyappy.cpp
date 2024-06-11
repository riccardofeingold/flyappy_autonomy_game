#include "flyappy_autonomy_code/agent/flyappy.hpp"

Flyappy::Flyappy() : 
    pid_(PIDController()),
    gateDetector_(GateDetection()),
    mpc_(MPC())
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
    Eigen::Matrix2d RMPC;
    // QMPC << 1200, 0, 0, 0,
    //      0, 500, 0, 0,
    //      0, 0, 1000, 0,
    //      0, 0, 0, 5;
    // RMPC << 0.001, 0,
    //      0, 0.001;
    QMPC << 1000, 0, 0, 0,
         0, 900, 0, 0,
         0, 0, 2000, 0,
         0, 0, 0, 15;
    RMPC << 0.00001, 0,
         0, 0.00001;
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

        XRef_ = Eigen::Vector4d(4.75, 0.5, 0, 0);
        if (stateEstimator_->getPosition().x() > 4.7)
        {
            stateChanged = true;
            currentState_ = States::TARGET;
        }
    } 
    else if (currentState_ == States::TUNNEL)
    {
        if (stateChanged)
        {
            std::cout << "STATE: TUNNEL" << std::endl;
            stateChanged = false;
        }
        
        // Switch to TARGET mode if we reach already safety distance
        double xref = gatePosition_.x() + wallWidth * 0.8;
                
        // Setting reference point
        XRef_ = Eigen::Vector4d(xref, VMAX-1, stateEstimator_->getPosition().y(), 0.0);
        
        if (stateEstimator_->getPosition().x() > xref)
        {
            stateChanged = true;
            currentState_ = States::TARGET;
        }

    } else if (currentState_ == States::TARGET)
    {
        if (stateChanged)
        {
            std::cout << "STATE: TARGET" << std::endl;
            stateChanged = false;
            explorePos_ = stateEstimator_->getPosition();
        }

        if (std::abs(gateDetector_.gate1->position.x() - gatePosition_.x()) > 1.5 && stateEstimator_->getPosition().x() > gateDetector_.gate1->position.x())
        {
            if (std::abs(stateEstimator_->getPosition().y() - gateDetector_.gate1->position.y()) < 0.03)
            {
                stateChanged = true;
                gatePosition_[0] = gateDetector_.closestPoints.closestPointWall1.x();
                gatePosition_[1] = gateDetector_.gate1->position.y();
                currentState_ = States::TUNNEL;
            }
        } else
        {   
            float deltaX = std::abs(stateEstimator_->getPosition().x() - gateDetector_.gate1->position.x());
            float velX = std::abs(explorePos_.y() - gateDetector_.gate1->position.y()) > HEIGHT_THRESHOLD ? deltaX/MAX_Y_SET_TIME : VMAX-1;
            velX = velX < VMAX/3 ? VMAX/3 : velX;

            if (std::abs(stateEstimator_->getPosition().y() - gateDetector_.gate1->position.y()) < 0.03)
            {
                velX = VMAX-1;
            }

            XRef_ = Eigen::Vector4d(gateDetector_.gate1->position.x(), velX, gateDetector_.gate1->position.y(), 0);
        }
    }
    
    // Track Path: Compute control input
    controlInput_ = Eigen::Vector2d::Zero();
    Eigen::VectorXd steadState = mpc_.computeSteadyState(XRef_);
    bool success = mpc_.solve(stateEstimator_->getStateVector().cast<double>(), steadState.segment(0, 4), steadState.segment(4, 2), controlInput_);
    if (!success) std::cout << "ERROR: NO U optimal found" << std::endl;
}

Eigen::Vector2d Flyappy::getControlInput()
{
    return controlInput_;
}