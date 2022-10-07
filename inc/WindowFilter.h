/*
 * WindowFilter.h
 *
 *  Created on: Jan 22, 2018
 *      Author: cui
 */

#ifndef WINDOWFILTER_H_
#define WINDOWFILTER_H_

#include <math.h>
#include <opencv2/video/video.hpp>

using namespace cv;
namespace lane {
class WindowFilter {
public:
  WindowFilter(float pos_init = 0.0, float meas_variance = 50,
               float process_variance = 1.0,
               float uncertainty_init = pow(2.0, 30));
  virtual ~WindowFilter();

public:
  void update(float x);
  float get_position() { return x_est; };
  void grow_uncertainty(int mag);
  float loglikelihood(float pos);

private:
  int state_dim;
  int meansure_dim;
  KalmanFilter *m_kalman;
  float x_est;
};
} // namespace lane

#endif /* WINDOWFILTER_H_ */
