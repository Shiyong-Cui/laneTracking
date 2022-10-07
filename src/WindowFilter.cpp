/*
 * WindowFilter.cpp
 *
 *  Created on: Jan 22, 2018
 *      Author: cui
 */

#include "WindowFilter.h"

#include <iostream>
#include <math.h>

using namespace std;
namespace lane {

WindowFilter::WindowFilter(float pos_init /* = 0.0*/,
                           float meas_variance /* = 50*/,
                           float process_variance /* = 1.0*/,
                           float uncertainty_init /* = pow(2.0, 30)*/)
    : state_dim(2), meansure_dim(1) {
  x_est = pos_init;
  m_kalman = new KalmanFilter(state_dim, meansure_dim);

  // set state transition function
  float F[] = {1, 1, 0, 0.5};
  m_kalman->transitionMatrix = Mat(state_dim, state_dim, CV_32FC1, F).clone();

  // Measurement function
  float H[] = {1.0, 0};
  m_kalman->measurementMatrix = Mat(1, 2, CV_32FC1, H).clone();

  // initial state
  m_kalman->statePost.at<float>(0) = pos_init;
  m_kalman->statePost.at<float>(1) = 0.0;

  // Initial co-variance matrix
  m_kalman->errorCovPost =
      Mat::eye(state_dim, state_dim, CV_32FC1) * uncertainty_init;

  // Measurement noise
  m_kalman->measurementNoiseCov =
      Mat::eye(meansure_dim, meansure_dim, CV_32FC1) * meas_variance;

  // Process noise
  float Q[] = {0.25, 0.5, 0.5, 1.0}; // generate random numbers
  m_kalman->processNoiseCov = Mat(2, 2, CV_32FC1, Q).clone();
}

WindowFilter::~WindowFilter() {
  if (m_kalman)
    delete m_kalman;
}

void WindowFilter::update(float x) {
  Mat measurement = Mat::zeros(1, 1, CV_32FC1);
  measurement.at<float>(0) = x;
  Mat pred_state = m_kalman->predict();
  Mat state = m_kalman->correct(measurement);
  x_est = state.at<float>(0);
}

void WindowFilter::grow_uncertainty(int mag) {
  for (int i = 0; i < mag; i++) {
    m_kalman->errorCovPost = m_kalman->transitionMatrix *
                                 m_kalman->errorCovPost *
                                 m_kalman->transitionMatrix.t() +
                             m_kalman->processNoiseCov;
  }
}

float WindowFilter::loglikelihood(float pos) {
  Mat temp = m_kalman->measurementMatrix * m_kalman->errorCovPost *
                 m_kalman->measurementMatrix.t() +
             m_kalman->measurementNoiseCov;
  float var = temp.at<float>(0);
  Mat_<float> state(2, 1);
  state.setTo(Scalar(0));
  state(0) = x_est;
  state(1) = 0.0;
  temp = m_kalman->measurementMatrix * state;
  float mean = temp.at<float>(0);
  float expo = (pos - mean) * (pos - mean) / 2.0 / var;
  float res = exp(expo) / sqrt(2 * M_PI * var);
  float logpdf = log(res);
  return logpdf;
}

} // namespace lane
