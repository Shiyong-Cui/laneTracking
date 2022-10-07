/*
 * LaneFinder.cpp
 *
 *  Created on: Jan 24, 2018
 *      Author: cui
 */

#include "LaneFinder.h"
#include <algorithm>
#include <map>
#include <opencv2/opencv.hpp>
#include <set>
#include <type_traits>

#include "PolynomialRegression.h"

namespace lane {
/**
 * @brief constructor
 *
 */
LaneFinder::LaneFinder(DashboardCamera *cam, vector<int> &window_shape,
                       int search_margin /*=200*/, int max_frozen_dur /*=15*/) {

  vector<int> img_shape;
  img_shape.push_back(cam->getImageHeight());
  img_shape.push_back(cam->getImageWidth());

  for (int i = 0; i < cam->getImageHeight() / window_shape[0]; i++) {
    int x_init_l = cam->getImageWidth() / 4;
    int x_init_r = cam->getImageWidth() * 3 / 4;

    windows_left.push_back(
        new Window(i, window_shape, img_shape, x_init_l, max_frozen_dur));
    windows_right.push_back(
        new Window(i, window_shape, img_shape, x_init_r, max_frozen_dur));
  }
  this->search_margin = search_margin;
  this->cam = cam;

  // color space setting
  settings.resize(3);
  settings[0].name = "lab_b";
  settings[0].cspace = "LAB";
  settings[0].channel = 2;
  settings[0].clipLimit = 2.0;
  settings[0].threshold = 150;

  settings[1].name = "value";
  settings[1].cspace = "HSV";
  settings[1].channel = 2;
  settings[1].clipLimit = 6.0;
  settings[1].threshold = 230; // 220;

  settings[2].name = "lightness";
  settings[2].cspace = "HLS";
  settings[2].channel = 1;
  settings[2].clipLimit = 2.0;
  settings[2].threshold = 200; // 210;

  clahe = cv::createCLAHE();
  clahe->setTilesGridSize(Size(8, 8));

  set<string> VIZ_OPTIONS = {"dash_undistorted",
                             "overhead",
                             "lab_b",
                             "lab_b_binary",
                             "lightness",
                             "lightness_binary",
                             "value",
                             "value_binary",
                             "pixel_scores",
                             "windows_raw",
                             "windows_filtered",
                             "highlighted_lane",
                             "presentation"};
}

/**
 * @brief Destroy the Lane Finder:: Lane Finder object
 *
 */
LaneFinder::~LaneFinder() {
  if (windows_left.size() != windows_right.size())
    cout << "windows_left and windows_right should have the same size." << endl;
  for (unsigned int i = 0; i < windows_left.size(); i++) {
    delete windows_left[i];
    delete windows_right[i];
  }
}

/**
 * @brief Takes a road image and returns an image where pixel intensity maps to
 likelihood of it being part of the lane.
 *  Each pixel gets its own score, stored as pixel intensity. An intensity of
 zero means it is not from the lane, and a higher score means higher confidence
 of being from the lane.
 * @param img an image of a road, typically from an overhead perspective. must
 be a RGB image
 * @return Mat the score image
*/
Mat LaneFinder::score_pixels(const Mat &img) {
  int w = img.cols, h = img.rows;
  Mat scores(h, w, CV_8UC1), temp;
  scores.setTo(Scalar(0));

  cvtColor(img, img, COLOR_BGR2RGB);
  vector<Mat> channels(3);
  string color_t;

  // #pragma omp parallel for
  for (unsigned int i = 0; i < settings.size(); i++) {

    color_t = "COLOR_RGB2" + settings[i].cspace;
    cvtColor(img, temp, getCode(color_t));
    split(temp, channels);
    Mat gray = channels[settings[i].channel];
    clahe->setClipLimit(settings[i].clipLimit);
    cv::Mat norm_img = gray.clone(); //
    clahe->apply(gray, norm_img);
    Mat binary;
    threshold(norm_img, binary, settings[i].threshold, 1.0, THRESH_BINARY);
    scores += binary;
  }
  Mat img_normalized;
  normalize(scores, img_normalized, 0, 255, NORM_MINMAX);
  return img_normalized;
}

/**
 * @brief find the code for color space transformation
 *
 * @param str a string name for the code
 * @return int the enum value for the code
 */
int LaneFinder::getCode(const string &str) {
  int code = COLOR_RGB2Lab;

  if (str == "COLOR_RGB2LAB")
    code = COLOR_RGB2Lab;
  else if (str == "COLOR_RGB2HSV")
    code = COLOR_RGB2HSV;
  else if (str == "COLOR_RGB2HLS")
    code = COLOR_RGB2HLS;

  return code;
}

/**
 * @brief compute the curvature
 *
 * @param windows
 * @return float the curvature
 */
float LaneFinder::calc_curvature(const vector<Window *> &windows) {
  int nb = windows.size();
  vector<float> x(nb, 0.0);
  vector<float> y(nb, 0.0);

  for (unsigned int i = 0; i < windows.size(); i++) {
    vector<int> xy = windows[i]->pos_xy();
    x.push_back(xy[0] * cam->getXMPerPix());
    y.push_back(xy[1] * cam->getYMPerPix());
  }

  int order = 2;
  vector<float> coeffs(order + 1, 0.0);
  PolynomialRegression<float> fitter;
  fitter.fitIt(y, x, order, coeffs);

  float y_eval = *max_element(y.begin(), y.end());
  float temp = 2 * coeffs[0] * y_eval * cam->getYMPerPix() + coeffs[1];
  temp *= temp;
  float curvature = pow(1 + temp, 1.5) / fabs(2.0 * coeffs[0]);

  return curvature;
}

/**
 * @brief Applies and returns a polynomial fit for given points along the left
 * and right lane line. Both lanes are described by a second order polynomial
 * x(y) = ay^2 + by + x0. In the `fit_globally` case, a and b are modeled as
 equal,
 * making the lines perfectly parallel. Otherwise, each line is fit independent
 of
 * the other. The parameters of the model are returned in a dictionary with
 * keys 'al', 'bl', 'x0l' for the left lane parameters and 'ar', 'br', 'x0r'
 * for the right lane.
 * @param points_left Two lists of the x and y positions along the left lane
 line.
 * @param points_right Two lists of the x and y positions along the right lane
 line.
 * @param fit_globally Set True to use the global, parallel line fit model. In
 practice this does not allays work.
 * @return map<string, float> a dictionary containing the fitting parameters for
 the left and right lane as above.
*/
map<string, float> LaneFinder::fit_lanes(const vector<Point> &points_left,
                                         const vector<Point> &points_right,
                                         bool fit_globally) {
  map<string, float> dict;
  vector<float> lx(points_left.size(), 0);
  vector<float> ly(points_left.size(), 0);
  vector<float> rx(points_right.size(), 0);
  vector<float> ry(points_right.size(), 0);

  // fill in the data
  for (unsigned int i = 0; i < points_left.size(); i++) {
    lx[i] = points_left[i].x;
    ly[i] = points_left[i].y;
  }
  for (unsigned int i = 0; i < points_right.size(); i++) {
    rx[i] = points_right[i].x;
    ry[i] = points_right[i].y;
  }

  if (fit_globally) {
    cout << "to be added ..." << endl;
  } else {
    int order = 2;
    PolynomialRegression<float> fitter;
    vector<float> coeffs_left; // (order + 1, 0.0);
    fitter.fitIt(ly, lx, order, coeffs_left);

    vector<float> coeffs_right; // (order + 1, 0.0);
    fitter.fitIt(ry, rx, order, coeffs_right);

    dict["x0l"] = coeffs_left[0];
    dict["bl"] = coeffs_left[1];
    dict["al"] = coeffs_left[2];

    dict["x0r"] = coeffs_right[0];
    dict["br"] = coeffs_right[1];
    dict["ar"] = coeffs_right[2];
  }
  return dict;
}

Mat LaneFinder::find_lines(const Mat &img_dashboard) {

  // Undistort and transform to overhead view
  img_overhead = cam->warp_to_overhead(img_dashboard);

  // Score pixels
  Mat half_img_overhead;
  resize(img_overhead, half_img_overhead, Size(0, 0), 0.5, 0.5, INTER_LINEAR);
  Mat half_pixel_scores = score_pixels(half_img_overhead);
  resize(half_pixel_scores, pixel_scores, Size(0, 0), 2, 2, INTER_LINEAR);
  // Select windows
  joint_sliding_window_update(windows_left, windows_right, pixel_scores,
                              search_margin);

  // Filter window positions
  vector<Window *> win_left_valid, win_right_valid;
  vector<int> argvalid_l, argvalid_r;
  filter_window_list(windows_left, win_left_valid, argvalid_l, false, true);
  filter_window_list(windows_right, win_right_valid, argvalid_r, false, true);

  if (win_left_valid.size() < 3 || win_right_valid.size() < 3) {
    cerr << "Not enough valid windows to create a fit." << endl;
    // Do something if not enough windows to fit. Most likely fall back on old
    // measurements.
    return Mat();
  }
  // collect data for lane fitting
  vector<Point> points_left(win_left_valid.size(), Point(0, 0)),
      points_right(win_right_valid.size(), Point(0, 0));
  for (unsigned int i = 0; i < win_left_valid.size(); i++) {
    win_left_valid[i]->pos_xy(points_left[i].x, points_left[i].y);
  }
  for (unsigned int i = 0; i < win_right_valid.size(); i++) {
    win_right_valid[i]->pos_xy(points_right[i].x, points_right[i].y);
  }
  // fit lanes
  map<string, float> fit_vals = fit_lanes(points_left, points_right);

  // Find a safe region to apply the polynomial fit over. We don't want to
  // extrapolate the shorter lane's extent.
  int short_line_max_ndx =
      min(argvalid_l[argvalid_l.size() - 1], argvalid_r[argvalid_r.size() - 1]);
  int nb = windows_left[0]->getYEnd() -
           windows_left[short_line_max_ndx]->getYBegin();

  // Determine the location of the polynomial fit line for each row of the image
  vector<int> y_fit(nb, 0);
  for (int i = windows_left[short_line_max_ndx]->getYBegin();
       i < windows_left[0]->getYEnd(); i++)
    y_fit[i - windows_left[short_line_max_ndx]->getYBegin()] = i;
  vector<float> x_fit_left(nb, 0.0), x_fit_right(nb, 0.0);
  for (int i = 0; i < nb; i++) {
    x_fit_left[i] = fit_vals["al"] * y_fit[i] * y_fit[i] +
                    fit_vals["bl"] * y_fit[i] + fit_vals["x0l"];
    x_fit_right[i] = fit_vals["ar"] * y_fit[i] * y_fit[i] +
                     fit_vals["br"] * y_fit[i] + fit_vals["x0r"];
  }

  // Calculate radius of curvature
  float curve_radius = calc_curvature(win_left_valid);

  // Calculate position in lane.
  int img_center = cam->getImageWidth() / 2.0;
  // img_center, [x_fit_left[-1], x_fit_right[-1]], [0, 1])
  vector<float> x(2, 0.0), y(2, 0.0);
  x[0] = *x_fit_left.end(), x[1] = *x_fit_right.end();
  y[0] = 0.0, y[1] = 1.0;
  vector<float> newx(1, 0.0);
  newx[0] = img_center;

  vector<float> res = interp1(x, y, newx);

  float lane_position_prcnt = res[0];
  float lane_position = lane_position_prcnt * REGULATION_LANE_WIDTH;

  Mat img_lane = viz_lane(img_dashboard, *cam, x_fit_left, x_fit_right, y_fit);
  return img_lane;
}

/**
 * @brief Updates Windows from both lists, preventing window crossover and
 * constraining their search regions to a margin. This improves on
 * `sliding_window_update()` by preventing windows from different lanes from
 * crossing over each other or detecting the same part of the image. Each
 * window's search region will be centered on the last undropped window position
 * and extend a margin to the left and right. In cases where the margins of the
 * left and right lane may overlap, they are truncated to the halfway point
 * between.
 * @param windows_left A list of Window objects for the left lane.
 * @param windows_right A list of Window objects for the right lane.
 * @param score_img A score image, where pixel intensity represents where the
 * lane
 * @param margin The maximum x distance the next window can be placed from the
 * last undropped window.
 */
void LaneFinder::joint_sliding_window_update(vector<Window *> &windows_left,
                                             vector<Window *> &windows_right,
                                             const Mat &score_img, int margin) {

  if (windows_left.size() != windows_right.size())
    cerr << "Window lists should be same length. Did you filter already?"
         << endl;
  if (windows_left.size() == 0)
    cerr << "no windows in the left list." << endl;
  if (windows_right.size() == 0)
    cerr << "no windows in the right list." << endl;

  vector<float> search_centers(2, 0.0);
  search_centers[0] = start_sliding_search(windows_left, score_img, 0);
  search_centers[1] = start_sliding_search(windows_right, score_img, 1);

  // Update each window, searching nearby the last undropped window.
  for (unsigned int i = 0; i < windows_left.size(); i++) {
    vector<float> x_search_ranges_left(2, 0.0), x_search_ranges_right(2, 0.0);
    x_search_ranges_left[0] = search_centers[0] - margin;
    x_search_ranges_left[1] = search_centers[0] + margin;

    x_search_ranges_right[0] = search_centers[1] - margin;
    x_search_ranges_right[1] = search_centers[1] + margin;

    // Fix any crossover
    if (x_search_ranges_left[1] > x_search_ranges_right[0]) {
      float middle = (x_search_ranges_left[1] + x_search_ranges_right[0]) / 2.0;
      x_search_ranges_left[1] = middle;
      x_search_ranges_right[0] = middle;
    }

    // Perform left update
    windows_left[i]->update(score_img, x_search_ranges_left);
    if (!windows_left[i]->droped())
      search_centers[0] = windows_left[i]->getFiltered();

    // Perform left update
    windows_right[i]->update(score_img, x_search_ranges_right);
    if (!windows_right[i]->droped())
      search_centers[1] = windows_right[i]->getFiltered();
  }
}

float LaneFinder::start_sliding_search(vector<Window *> &windows,
                                       const Mat &score_img, int mode) {
  if (windows.size() == 0)
    cerr << "no windows in start_sliding_searc." << endl;
  // mode is either 0 = left or 1 = right
  if (mode != 0 && mode != 1)
    cerr << "mode must be either 0 or 1." << endl;
  vector<float> y(windows.size(), 0.0);
  for (unsigned int i = 0; i < windows.size(); i++)
    y[i] = windows[i]->getY();
  bool bd = strictly_decreasing(y);
  if (!bd)
    cerr << "Windows not ordered properly. Should start at image bottom. "
         << endl;
  int img_h = score_img.rows, img_w = score_img.cols;
  // Update the bottom window
  vector<float> search_range(2, 0.0);
  if (mode == 0) { // left
    search_range[0] = 0;
    search_range[1] = floor(img_w / 2);
  } else { // right
    search_range[0] = floor(img_w / 2);
    search_range[1] = img_w;
  }
  // update the first window from bottom either left side or right side
  windows[0]->update(score_img, search_range);

  float search_center = 0.0;
  // Find the starting point for our search
  if (windows[0]->droped()) {
    // Starting window does not exist, find an approximation.
    // search bottom 1/3rd of score_img
    Mat search_region =
        score_img.rowRange(int(floor(2.0 * img_h / 3.0)), img_h);

    // apply gaussian filter to the column sum
    Mat_<float> col_sum(1, search_region.cols);
    col_sum.setTo(Scalar(0));
    Mat_<float> column_scores(1, search_region.cols);
    column_scores.setTo(Scalar(0));
    // column sum
    cv::reduce(search_region, col_sum, 0, REDUCE_SUM, CV_32F);

    // Gaussian filter
    float truncate = 3.0;
    float sd = windows[0]->getWidth() / 5.0;
    int ksize = int(truncate * sd + 0.5);
    GaussianBlur(col_sum, column_scores, cv::Size(1, ksize), BORDER_CONSTANT,
                 0);

    vector<float> vec;
    column_scores.row(0).copyTo(vec);

    if (mode == 0) {
      search_center = argmax_between(vec, 0, int(floor(img_w / 2.0)));
    } else {
      search_center = argmax_between(vec, int(floor(img_w / 2.0)), img_w);
    }
  } else
    search_center = windows[0]->getFiltered();

  return search_center;
}

bool LaneFinder::strictly_decreasing(vector<float> &list) {
  int nb = list.size();
  for (int i = 0; i < nb - 1; i++) {
    if (list[i] < list[i + 1])
      return false;
  }
  return true;
}

void LaneFinder::filter_window_list(const vector<Window *> &windows,
                                    vector<Window *> &windows_filtered,
                                    vector<int> &indexes, bool remove_frozen,
                                    bool remove_dropped,
                                    bool remove_undetected) {
  if (!windows.size())
    return;
  if (windows_filtered.size())
    windows_filtered.clear();
  if (indexes.size())
    indexes.clear();
  for (unsigned int i = 0; i < windows.size(); i++) {
    if (windows[i]->droped() && remove_dropped)
      continue;
    if (windows[i]->isFrozen() && remove_frozen)
      continue;
    if (!windows[i]->isDetected() && remove_undetected)
      continue;

    windows_filtered.push_back(windows[i]);
    indexes.push_back(i);
  }
}

int LaneFinder::argmax_between(const vector<float> &values, int begin,
                               int end) {
  int siz = values.size();
  if (begin < 0 || begin > siz)
    cout << "index begin should be within [0, siz)" << endl;
  if (end < 0 || end > siz)
    cout << "index end should be within [0, siz)" << endl;
  if (begin >= end)
    cout << "begin > end in LaneFinder::argmax_between" << endl;
  int mx_id = begin;
  float mx = values[0];
  for (int i = begin + 1; i < end; i++) {
    if (mx < values[i]) {
      mx = values[i];
      mx_id = i;
    }
  }
  if (mx <= 0.0)
    mx_id = (begin + end) / 2.0;
  return mx_id;
}

Mat LaneFinder::window_image(const vector<Window *> &windows, int opt,
                             const Scalar color, const Scalar color_frozen,
                             const Scalar color_droped) {

  if (windows.size() == 0)
    cout << "no windows in LaneFinder::window_image." << endl;
  int img_h = windows[0]->getImgH(), img_w = windows[0]->getImgW();
  Mat mask(img_h, img_w, CV_8UC3);
  mask.setTo(Scalar(0, 0, 0));
  if (windows.size() <= 0) {
    cout << " no windows in LaneFinder::window_image. " << endl;
    return mask;
  }

  Scalar color_curr;
  int r = 0, c = 0;
  for (unsigned int i = 0; i < windows.size(); i++) {
    if (windows[i]->droped())
      color_curr = color_droped;
    else if (windows[i]->isFrozen())
      color_curr = color_frozen;
    else
      color_curr = color;
    Mat temp = windows[i]->get_mask(opt);
    for (int j = 0; j < img_h * img_w; j++) {
      r = j % img_w;
      c = j - r * img_w;
      unsigned char pix = temp.at<unsigned char>(r, c);
      if (pix > 0) {
        Vec3b &color = mask.at<Vec3b>(r, c);
        color[0] = color_curr[0];
        color[1] = color_curr[1];
        color[2] = color_curr[2];
      }
    }
  }
  return mask;
}

template <typename Real>
int LaneFinder::nearestNeighbourIndex(std::vector<Real> &x, Real &value) {
  Real dist = std::numeric_limits<Real>::max();
  Real newDist = dist;
  size_t idx = 0;

  for (size_t i = 0; i < x.size(); ++i) {
    newDist = std::abs(value - x[i]);
    if (newDist <= dist) {
      dist = newDist;
      idx = i;
    }
  }

  return idx;
}

template <typename Real>
std::vector<Real> LaneFinder::interp1(std::vector<Real> &x,
                                      std::vector<Real> &y,
                                      std::vector<Real> &x_new) {
  std::vector<Real> y_new;
  Real dx, dy, m, b;
  size_t x_max_idx = x.size() - 1;
  size_t x_new_size = x_new.size();

  y_new.reserve(x_new_size);

  for (size_t i = 0; i < x_new_size; ++i) {
    size_t idx = nearestNeighbourIndex(x, x_new[i]);
    if (x[idx] > x_new[i]) {
      dx = idx > 0 ? (x[idx] - x[idx - 1]) : (x[idx + 1] - x[idx]);
      dy = idx > 0 ? (y[idx] - y[idx - 1]) : (y[idx + 1] - y[idx]);
    } else {
      dx = idx < x_max_idx ? (x[idx + 1] - x[idx]) : (x[idx] - x[idx - 1]);
      dy = idx < x_max_idx ? (y[idx + 1] - y[idx]) : (y[idx] - y[idx - 1]);
    }
    m = dy / dx;
    b = y[idx] - x[idx] * m;
    y_new.push_back(x_new[i] * m + b);
  }

  return y_new;
}

Mat LaneFinder::viz_windows(const Mat &score_img, int mode) {
  // Displays the position of the windows over a score image.
  Mat lw_img, rw_img;
  if (mode == 0) { // 'filtered':
    lw_img = window_image(windows_left, 0, Scalar(0, 255, 0));
    rw_img = window_image(windows_right, 0, Scalar(0, 255, 0));
  } else if (mode == 1) { // 'raw':
    vector<Window *> win_left_detected, win_right_detected;
    vector<int> arg_left, arg_right;
    Scalar color = Scalar(0, 0, 255);
    filter_window_list(windows_left, win_left_detected, arg_left, false, false,
                       true);
    filter_window_list(windows_right, win_right_detected, arg_right, false,
                       false, true);
    lw_img = window_image(win_left_detected, 1, color, color, color);
    rw_img = window_image(win_right_detected, 1, color, color, color);
  } else
    cerr << "mode is not valid." << endl;

  Mat combined = lw_img + rw_img;
  Mat res;
  Mat color;
  cv::cvtColor(score_img, color, cv::COLOR_GRAY2BGR);
  addWeighted(color, 1, combined, 0.5, 0.0, res);
  return res;
}

/**
 * @brief Take an undistorted dashboard camera image and highlights the lane.
 *
 * @param undist_img An undistorted dashboard view image.
 * @param camera The DashboardCamera object for the camera the image was taken
 * on.
 * @param left_fit_x the x values for the left line polynomial at the given y
 * values
 * @param right_fit_x the x values for the right line polynomial at the given y
 * values
 * @param fit_y the y values the left and right line x values were calculated at
 * @return Mat The undistorted image with the lane overlaid on top of it.
 */
Mat LaneFinder::viz_lane(const Mat &undist_img, const DashboardCamera &camera,
                         const vector<float> &left_fit_x,
                         const vector<float> &right_fit_x,
                         const vector<int> &fit_y) {

  // Create an undist_img to draw the lines on
  Mat lane_poly_overhead(undist_img.size(), CV_8UC3, Scalar(0, 0, 0));

  //
  int nbp = left_fit_x.size() + right_fit_x.size();
  vector<Point> polygon(nbp);
  for (unsigned int i = 0; i < left_fit_x.size(); i++) {
    polygon[i].x = left_fit_x[i];
    polygon[i].y = fit_y[i];
  }

  for (unsigned int i = 0; i < right_fit_x.size(); i++) {
    polygon[i + left_fit_x.size()].x = right_fit_x[right_fit_x.size() - 1 - i];
    polygon[i + left_fit_x.size()].y = fit_y[right_fit_x.size() - 1 - i];
  }

  vector<vector<Point>> PPoint;
  PPoint.push_back(polygon);
  fillPoly(lane_poly_overhead, PPoint, Scalar(0, 255, 0));

  // Warp back to original undist_img space
  Mat lane_poly_dash = cam->warp_to_dashboard(lane_poly_overhead);

  // Combine the result with the original undist_img
  Mat res;
  addWeighted(undist_img, 1, lane_poly_dash, 0.3, 0.0, res);

  return res;
}

} // namespace lane
