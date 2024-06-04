#pragma once

// Environment knowledge
const double SAMPLING_TIME = 1.0/30.0;
constexpr float pixelInMeters = 0.01f;
const float wallWidth = 0.6;
const float gateHeight = 0.4;
const float pipeGap = 1.9;
const float birdHeight = 24 * pixelInMeters;

// constraints
const float VMIN = -4.5;
const float VMAX = 4.5;
const float MAX_Y_SET_TIME = 1.0f;
const float HEIGHT_THRESHOLD = 0.5;
const float X_SAFE_MARGIN = 0.1;
const float axUpperBound = 3;
const float axLowerBound = -3;
const float ayUpperBound = 35;
const float ayLowerBound = -35;

// States
enum States {
  INIT,
  MOVE_FORWARD,
  TUNNEL,
  TARGET
};

enum ResetStates
{
    CLEAR_ALL,
    CLEAR_OLDEST,
    CLEAR_ALL_NEAR_CLOSEST_POINT
};