#pragma once

#include <iostream>
#include <vector>
#include <deque>
#include <numeric>

class KalmanFilter {
public:
    KalmanFilter(double process_noise, double measurement_noise, double estimation_error, double initial_value)
    {
        Q = process_noise;  // Process noise covariance
        R = measurement_noise;  // Measurement noise covariance
        P = estimation_error;  // Estimation error covariance
        x = initial_value;  // Initial estimate
    }

    void predict() {
        // Prediction step
        // x = A * x; since A is 1, x remains x
        // P = A * P * A^T + Q; since A is 1, P = P + Q
        P += Q;
    }

    void update(double measurement) {
        // Update step
        double K = P / (P + R);  // Kalman gain
        x = x + K * (measurement - x);  // Update estimate
        P = (1 - K) * P;  // Update error covariance
    }

    double getEstimate() const {
        return x;
    }

private:
    double Q;  // Process noise covariance
    double R;  // Measurement noise covariance
    double P;  // Estimation error covariance
    double x;  // Value
};