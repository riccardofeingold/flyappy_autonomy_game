#pragma once

// Environment knowledge
const double SAMPLING_TIME = 1.0/30.0;
constexpr float pixelInMeters = 0.01f;
const float wallWidth = 0.9;
const float gateHeight = 0.5;
const float pipeGap = 2.0;
const float birdHeight = 24 * pixelInMeters;

// constraints
const float axUpperBound = 3;
const float axLowerBound = -3;
const float ayUpperBound = 35;
const float ayLowerBound = -35;

// States
enum States {
  INIT,
  MOVE_FORWARD,
  EXPLORE,
  TARGET
};

enum ResetStates
{
    CLEAR_ALL,
    CLEAR_OLDEST,
    CLEAR_ALL_NEAR_CLOSEST_POINT
};