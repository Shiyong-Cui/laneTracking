/*
 * LaneFinder.h
 *
 *  Created on: Jan 24, 2018
 *      Author: cui
 */

#ifndef LANEFINDER_H_
#define LANEFINDER_H_

#include <vector>
#include <map>
#include "DashboardCamera.h"
#include "Window.h"
// #include "CLAHEAlgorithm.h"

using namespace std;
using namespace cv;
using namespace lane;

namespace lane {

// Settings to run thresholding operations on
struct setting {
	string name;
	string cspace;
	int channel;
	double clipLimit;
	double threshold;
};


class LaneFinder {
public:
	LaneFinder(DashboardCamera* cam, vector<int>& window_shape, int search_margin=90, int max_frozen_dur=15);
	virtual ~LaneFinder();

	Mat score_pixels(const Mat& img);
	int getCode(const string& str);
	float calc_curvature(const vector<Window*>& windows);
	map<string, float> fit_lanes(const vector<Point>& points_left, const vector<Point>& points_right, bool fit_globally = false);
	Mat find_lines(const Mat& img_dashboard);

	// static functions
	static void joint_sliding_window_update(vector<Window*>& windows_left, vector<Window*>& windows_right, const Mat& score_img, int margin);
	static float start_sliding_search(vector<Window*>& windows, const Mat& score_img, int mode = 0);
	static bool strictly_decreasing(vector<float>& list);
	static void filter_window_list(const vector<Window*>& windows, vector<Window*>& windows_filtered, vector<int>& indexes, \
			bool remove_frozen = false, bool remove_dropped = true, bool remove_undetected = false);
	static int argmax_between(const vector<float>& values, int begin, int end);
	void normalizeMat(const Mat& src, Mat& dest, float min, float max);


	template<typename Real>
	int nearestNeighbourIndex(std::vector<Real> &x, Real &value);
	template<typename Real>
	std::vector<Real> interp1(std::vector<Real> &x, std::vector<Real> &y, std::vector<Real> &x_new);
	void Test_Interp1();

	// functions for visualization
	// Mat viz_presentation(const Mat& lane_img, lane_position, float curve_radius, float lane_width=REGULATION_LANE_WIDTH)
	Mat viz_windows(const Mat& score_img, int mode);
	Mat viz_lane(const Mat& undist_img, const DashboardCamera& camera, const vector<float>& left_fit_x, const vector<float>& right_fit_x, const vector<int>& fit_y);

	static Mat window_image(const vector<Window*>& windows, int opt = 0, const Scalar color = Scalar( 0, 255, 0 ), \
			const Scalar color_frozen = 0.2 * Scalar( 0, 255, 0 ), const Scalar color_droped = Scalar(0, 0, 0) );

	const Mat& getImgDashUndistorted() const { return img_dash_undistorted; }
	const Mat& getImgFilteredWin() const { 	return img_filtered_win; }
	const Mat& getImgOverhead() const { return img_overhead; }
	const Mat& getImgRawWin() const { return img_raw_win; }
	const Mat& getPixelScores() const { return pixel_scores; }

private:
	const float REGULATION_LANE_WIDTH = 3.7;
	lane::DashboardCamera* cam;
	vector<Window*> windows_left;
	vector<Window*> windows_right;
	int search_margin;
	map<string, Mat> visuals;

	// for debugging, should be removed for production
	void display(const Mat& img);
	Mat img_dash_undistorted;
	Mat img_overhead;
	Mat pixel_scores;
	Mat img_filtered_win;
	Mat img_raw_win;

	// for color space
	vector<setting> settings;

	//
	cv::Ptr<cv::CLAHE> clahe;
	// CLAHEAlgorithm* pAlgo_CLAHE;


private:
	// functions for contrast enhancement
	Mat eaualizeHist_GO(Mat src);
	Mat aheGO(Mat src,int _step = 8);
	Mat clheGO(Mat src,int _step = 8);
	Mat claheGoWithoutInterpolation(Mat src, int _step = 8);
	Mat claheGO(Mat src,int _step = 8);


};

} /* namespace Lane */

#endif /* LANEFINDER_H_ */
