/*
 * DashboardCamera.cpp
 *
 *  Created on: Jan 18, 2018
 *      Author: cui
 */

#include "DashboardCamera.h"
#include "Timer.h"
#include <iostream>

namespace lane {

DashboardCamera::DashboardCamera(const string &file, int image_height,
                                 int img_width,
                                 const vector<cv::Point2f> &lane_shape,
                                 const vector<float> &scale_correction) {

  m_ImgWidth = img_width;
  m_ImgHeight = image_height;

  // load calibration parameters
  loadCameraParameters(file);

  // compute perspective transformation
  computeTransform(lane_shape, scale_correction);
}

DashboardCamera::DashboardCamera(
    vector<string> chessboard_img_fnames, vector<int> chessboard_size,
    vector<cv::Point2f> lane_shape,
    vector<float> scale_correction /*={30 / 720.0, 3.7 / 700}*/) {
  if (chessboard_img_fnames.size() < 1)
    cout << "no chess board images for calibration." << endl;
  Mat image = imread(chessboard_img_fnames[0], IMREAD_COLOR); // Read the file
  m_ImgWidth = image.cols;
  m_ImgHeight = image.rows;

  // calibrate the camera
  calibrate(chessboard_img_fnames, chessboard_size);

  computeTransform(lane_shape, scale_correction);
}

DashboardCamera::~DashboardCamera() {}

void DashboardCamera::calibrate(vector<string> chessboard_img_fnames,
                                vector<int> chessboard_size) {

  int chess_rows = chessboard_size[0];
  int chess_cols = chessboard_size[1];

  cv::Size board_sz = cv::Size(chess_cols, chess_rows);

  vector<vector<cv::Point2f>> image_points;
  vector<vector<cv::Point3f>> object_points;

  vector<cv::Point3f> objectCorners;

  for (int i = 0; i < chess_rows; i++) {
    for (int j = 0; j < chess_cols; j++) {
      objectCorners.push_back(cv::Point3f(j, i, 0.0f));
    }
  }

  cv::Size image_size;
  for (unsigned int i = 0; i < chessboard_img_fnames.size(); i++) {
    // Read the file
    Mat image = cv::imread(chessboard_img_fnames[i], IMREAD_COLOR);

    // down sample the image
    Mat half_imgae;
    resize(image, half_imgae, Size(0, 0), 0.5, 0.5, INTER_CUBIC);

    // convert to gray scale image
    Mat imageGray;
    cvtColor(half_imgae, imageGray, COLOR_BGR2GRAY);
    image_size = imageGray.size();

    vector<cv::Point2f> imageCorners;
    Timer timer;
    timer.start();
    bool found = findChessboardCorners(imageGray, board_sz, imageCorners);
    timer.stop();
    double t = timer.print("coner: ");

    if (found) {
      cout << chessboard_img_fnames[i] << "  True" << endl;

      // Get subpixel accuracy on the corners
      cornerSubPix(
          imageGray, imageCorners, cv::Size(5, 5), cv::Size(-1, -1),
          cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS,
                           30,    // max number of iterations
                           0.1)); // min accuracy

      image_points.push_back(imageCorners);
      object_points.push_back(objectCorners);
    } else {
      cout << chessboard_img_fnames[i] << "  False" << endl;
    }
  }

  // Output rotations and translations
  vector<Mat> rvecs, tvecs;
  double RMS = calibrateCamera(object_points, image_points, image_size,
                               m_camera_matrix, m_distortion_coeffs, rvecs,
                               tvecs); //,	cv::CALIB_ZERO_TANGENT_DIST |
                                       //cv::CALIB_FIX_PRINCIPAL_POINT	);

  cout << "RMS error is " << RMS << endl;

  cout << "Camera matrix: " << endl;
  cout << m_camera_matrix << endl;
  cout << "Distortion coefficients: " << endl;
  cout << m_distortion_coeffs << endl;
}

Mat DashboardCamera::undistort(const Mat &image) {
  Mat imageUndistorted;
  cv::undistort(image, imageUndistorted, m_camera_matrix, m_distortion_coeffs);
  return imageUndistorted;
}

Mat DashboardCamera::warp_to_overhead(const Mat &image) {
  Mat warped_image;
  remap(image, warped_image, m_mapx, m_mapy, INTER_NEAREST);

  return warped_image;
}

Mat DashboardCamera::warp_to_dashboard(const Mat &image) {
  Mat over_head_image;

  remap(image, over_head_image, m_inv_mapx, m_inv_mapy, INTER_NEAREST);
  return over_head_image;
}

void DashboardCamera::saveCameraParameters(const string &file) {

  // write Mat to file
  cv::FileStorage fs(file, cv::FileStorage::WRITE);
  fs << "camera matrix" << m_camera_matrix;
  fs << "distortion coefficients" << m_distortion_coeffs;
  fs.release();
}

void DashboardCamera::computeTransform(const vector<cv::Point2f> &lane_shape,
                                       const vector<float> &scale_correction) {
  // compute perspective transformation
  cv::Point2f src_points[] = {lane_shape[0], lane_shape[1], lane_shape[3],
                              lane_shape[2]};

  // [top_left, top_right, bottom_left, bottom_right]
  cv::Point2f dst_points[] = {cv::Point2f(lane_shape[2].x, 0),
                              cv::Point2f(lane_shape[3].x, 0),
                              cv::Point2f(lane_shape[3].x, m_ImgHeight - 1),
                              cv::Point2f(lane_shape[2].x, m_ImgHeight - 1)};

  m_overhead_transform = cv::getPerspectiveTransform(src_points, dst_points);
  m_inverse_overhead_transform =
      cv::getPerspectiveTransform(dst_points, src_points);

  perspective_to_maps(m_overhead_transform, cv::Size(m_ImgWidth, m_ImgHeight),
                      m_mapx, m_mapy);
  perspective_to_maps(m_inverse_overhead_transform,
                      cv::Size(m_ImgWidth, m_ImgHeight), m_inv_mapx,
                      m_inv_mapy);

  m_x_m_per_pix = scale_correction[0];
  m_y_m_per_pix = scale_correction[1];
}

void DashboardCamera::loadCameraParameters(const string &file) {

  // load from file
  cv::FileStorage fs(file, cv::FileStorage::READ);
  fs["camera matrix"] >> m_camera_matrix;
  fs["distortion coefficients"] >> m_distortion_coeffs;
  fs.release();
}

void DashboardCamera::perspective_to_maps(const cv::Mat &perspective_mat,
                                          const cv::Size &img_size,
                                          cv::Mat &mapx, cv::Mat &mapy) {
  // invert the matrix because the transformation maps must be
  // bird's view -> original
  cv::Mat inv_perspective(perspective_mat.inv());
  inv_perspective.convertTo(inv_perspective, CV_32FC1);

  // create XY 2D array
  // (((0, 0), (1, 0), (2, 0), ...),
  //  ((0, 1), (1, 1), (2, 1), ...),
  // ...)
  cv::Mat xy(img_size, CV_32FC2);
  float *pxy = (float *)xy.data;
  for (int y = 0; y < img_size.height; y++)
    for (int x = 0; x < img_size.width; x++) {
      *pxy++ = x;
      *pxy++ = y;
    }

  // perspective transformation of the points
  cv::Mat xy_transformed;
  cv::perspectiveTransform(xy, xy_transformed, inv_perspective);

  // split x/y to extra maps
  assert(xy_transformed.channels() == 2);
  cv::Mat maps[2]; // map_x, map_y
  cv::split(xy_transformed, maps);

  // remap() with integer maps is faster
  cv::convertMaps(maps[0], maps[1], mapx, mapy, CV_16SC2);
}

} // namespace lane
