/*
 * DashboardCamera.h
 *
 *  Created on: Jan 18, 2018
 *      Author: cui
 */

#ifndef DASHBOARDCAMERA_H_
#define DASHBOARDCAMERA_H_

#include <vector>
#include <opencv2/opencv.hpp>



using namespace cv;
using namespace std;

namespace lane {




class DashboardCamera {
public:
	// create camera from images prepared for calibration
	// lane_shape: [top_left, top_right, bottom_left, bottom_right]
	DashboardCamera(vector<string> chessboard_img_fnames, vector<int> chessboard_size, vector<cv::Point2f> lane_shape, \
			vector<float> scale_correction/*={30 / 720.0, 3.7 / 700}*/);
	// load a camera from a file
	DashboardCamera(const string& file, int image_height, int img_width, const vector<cv::Point2f>& lane_shape, const vector<float>& scale_correction);
	virtual ~DashboardCamera();

	// Removes distortion this camera's raw images.
	Mat undistort(const Mat& image);
	// Transforms this camera's images from the dashboard perspective to an overhead perspective.
	Mat warp_to_overhead(const Mat& image);
	// Transforms this camera's images from an overhead perspective back to the dashboard perspective.
	Mat warp_to_dashboard(const Mat& image);

	void saveCameraParameters(const string& file);
	void loadCameraParameters(const string& file);
	// geters
	int getImageHeight() { return m_ImgHeight;};
	int getImageWidth() {
		return m_ImgWidth;
	}

	float getXMPerPix() const {
		return m_x_m_per_pix;
	}

	void setXMPerPix(float xMPerPix) {
		m_x_m_per_pix = xMPerPix;
	}

	float getYMPerPix() const {
		return m_y_m_per_pix;
	}

	void setYMPerPix(float yMPerPix) {
		m_y_m_per_pix = yMPerPix;
	}

	;


protected:
	// Calibrates the camera using chessboard calibration images.
	void calibrate(vector<string> chessboard_img_fnames, vector<int> chessboard_size);
	void computeTransform(const vector<cv::Point2f>& lane_shape, const vector<float>& scale_correction);

	// compute perspective maps
	void perspective_to_maps(const cv::Mat &perspective_mat, const cv::Size &img_size, cv::Mat &mapx, cv::Mat &mapy);



private:
	int m_ImgWidth;
	int m_ImgHeight;

	// pixel size in birds' eye view
	float m_x_m_per_pix;
	float m_y_m_per_pix;

	Mat m_overhead_transform;
	Mat m_inverse_overhead_transform;

	// to avoid using warpPerspective
	Mat m_mapx;
	Mat m_mapy;
	Mat m_inv_mapx;
	Mat m_inv_mapy;




	Mat m_camera_matrix;
	Mat m_distortion_coeffs;
};

}

#endif /* DASHBOARDCAMERA_H_ */
