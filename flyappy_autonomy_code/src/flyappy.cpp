#include "flyappy_autonomy_code/flyappy.hpp"

Flyappy::Flyappy() : 
    pid_(PIDController()),
    gateDetector_(GateDetection())
{
    stateEstimator_ = std::make_shared<StateEstimation>(SAMPLING_TIME);
    velMeasured_ = Eigen::Vector2f::Zero();
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
        controlInput_ = pid_.computeAcceleration(Eigen::Vector2f(0, 0.1), Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());
    } 
    else
    {
        gateDetector_.computeBoundaries();
        gateDetector_.initDone = true;
        controlInput_ = pid_.computeAcceleration(Eigen::Vector2f::Zero(), Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());

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
        std::cout << "STATE: MOVE_FORWARD" << std::endl;
        if (stateChanged)
        {
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
        // if (gateDetector_.gate->gateUpdated)
        // {
        //     if (std::abs(stateEstimator_->getPosition().y() - gateDetector_.gate->position.y()) < 0.01)
        //     {
        //         controlInput_ = pid_.computeAcceleration(gateDetector_.gate->position + Eigen::Vector2f(1.5, 0), Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());
        //     } else
        //     {
        //         controlInput_ = pid_.computeAcceleration(Eigen::Vector2f(0, gateDetector_.gate->position.y()), Eigen::Vector2f::Zero(), stateEstimator_->getPosition(), stateEstimator_->getVelocity());
        //     }
        // }
    }
}

Eigen::Vector2f Flyappy::getControlInput()
{
    return controlInput_;
}