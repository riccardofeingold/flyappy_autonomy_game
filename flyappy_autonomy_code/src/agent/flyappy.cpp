#include "flyappy_autonomy_code/agent/flyappy.hpp"

Flyappy::Flyappy() : 
    pid_(PIDController()),
    gateDetector_(GateDetection())
{
    stateEstimator_ = std::make_shared<StateEstimation>(SAMPLING_TIME);
    velMeasured_ = Eigen::Vector2f::Zero();
    
    // initialize the LQR Controller
    Eigen::Matrix4f Q;
    Q << 100, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 100, 0,
         0, 0, 0, 1;
    Eigen::Matrix2f R;
    R << 200, 0,
         0, 1;

    // calculated using matlab
    Eigen::Matrix<float, 2, 4> K;
    // K << 0.6932, 1.1795, 0, 0,
    //      0, 0, 9.2648, 4.4032;
    K << 1.375, 1.664, 0, 0,
         0, 0, 15.6807, 5.6728;
    lqr_ = LQR(Q, R, 1000, K);
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
        controlInput_ = lqr_.eval(stateEstimator_->getStateVector() - XRef);
        // controlInput_ = pid_.computeAcceleration(Eigen::Vector2f(0, 0.5), Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());
    } 
    else
    {
        gateDetector_.computeBoundaries();
        gateDetector_.initDone = true;
        controlInput_ = lqr_.eval(stateEstimator_->getStateVector());
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
        
        Eigen::Vector2f refPos(2, 0);
        controlInput_ = pid_.computeAcceleration(refPos, Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());

        if (stateEstimator_->getVelocity().norm() < 0.005 && stateEstimator_->getPosition().x() > refPos.x())
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
        
        // Motion Planning
        // pathPlanner_->update(stateEstimator_->getPosition(), stateEstimator_->getVelocity());
        // Eigen::Vector2f ref_pos = pathPlanner_->getNextRefPoint();

        // Track Path:
        controlInput_ = Eigen::Vector2f::Zero();
        if (std::abs(stateEstimator_->getPosition().y() - gateDetector_.gate->position.y()) < 0.01)
        {
            Eigen::Vector4f XRef = Eigen::Vector4f(gateDetector_.gate->position.x() + 2.5, 0, gateDetector_.gate->position.y(), 0);
            controlInput_ = lqr_.eval(stateEstimator_->getStateVector() - XRef);
            // controlInput_ = pid_.computeAcceleration(gateDetector_.gate->position + Eigen::Vector2f(0.8, 0), Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());
        } else
        {
            Eigen::Vector4f XRef = Eigen::Vector4f(0, 0, gateDetector_.gate->position.y(), 0);
            controlInput_ = lqr_.eval(stateEstimator_->getStateVector() - XRef);
            // controlInput_ = pid_.computeAcceleration(Eigen::Vector2f(0, gateDetector_.gate->position.y()), Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());
        }
    }
}

Eigen::Vector2f Flyappy::getControlInput()
{
    return controlInput_;
}