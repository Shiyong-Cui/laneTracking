/*
 * Window.cpp
 *
 *  Created on: Jan 24, 2018
 *      Author: cui
 */


#include <iostream>
#include "Window.h"

using namespace std;

namespace lane {



Window::Window(int level, vector<int> window_shape, vector<int> img_shape, float x_init, int max_frozen_dur) {

	if (img_shape.size() < 2) cerr << "no image shape" << endl;
	if (window_shape.size() < 2) cerr << "no window shape" << endl;
    if (window_shape[1] % 2 == 0) cerr << "width must be odd" << endl;
    // Image info
    img_h = img_shape[0];
    img_w = img_shape[1];

    height = window_shape[0];
    width = window_shape[1];
    y_begin = img_h - (level + 1) * height;  // top row of pixels for window
    y_end = y_begin + height;  // one past the bottom row of pixels for window

    x_filtered = x_init;
    y = y_begin + height / 2.0;
    this->level = level;

    filter = new WindowFilter(x_init);
    x_measured = 0;
    detected = false;

    this->max_frozen_dur = max_frozen_dur;
    frozen = false;
    frozen_dur = max_frozen_dur + 1;
    // frozen_dur = 0;
    undrop_buffer = 1;
}

Window::~Window() {
	if (filter) delete filter;
}

int Window::x_begin(int opt/* = 0*/)
{
	int x = (opt == 0) ? x_filtered : x_measured;
	int res = int(max(0, x - width / 2));
	return res;
}

int Window::x_end(int opt /* = 0*/)
{
	int x = (opt == 0) ? x_filtered : x_measured;
	int res = int(min(x + width / 2, img_w));
	return res;
}


vector<int> Window::pos_xy(int opt /*= 0*/)
{
	int x = (opt == 0) ? x_filtered : x_measured;
	vector<int> pos(2, 0);
	pos[0] = x; pos[1] = y;
	return pos;
}

void Window::pos_xy(int& x, int& y, int opt) {
	x = (opt == 0) ? x_filtered : x_measured;
	y = this->y;
}

void Window::freeze()
{
    frozen = true;
    frozen_dur += 1;
    filter->grow_uncertainty(1);
}

void Window::unfreeze()
{
    frozen_dur = min(frozen_dur, max_frozen_dur + 1 + undrop_buffer);
    frozen_dur -= 1;
    frozen_dur = max(0, frozen_dur);

    // Change states
    frozen = false;
}

Mat Window::get_mask(int opt /*= 0*/)
{
	int x_start = x_begin(opt);
	int x_last = x_end(opt);
	Mat mask( img_h, img_w, CV_8UC1);
	mask.setTo(Scalar(0));
	mask( cv::Range(y_begin, y_end), cv::Range(x_start, x_last)) = 1;
	return mask;
}

void Window::update(const Mat& score_img, vector<float>& x_search_range, float min_log_likelihood /*=-40*/ )
{
	/*  Given a score image and the x search bounds, updates the window position to the likely position of the lane.

        If the measurement is deemed suspect for some reason, the update will be rejected and the window will be
        'frozen', causing it to stay in place. If the window is frozen beyond its  `max_frozen_dur` then it will be
        dropped entirely until a non-suspect measurement is made.

        The window only searches within its y range defined at initialization.

        :param score_img: A score image, where pixel intensity represents where the lane most likely is.
        :param x_search_range: The (x_begin, x_end) range the window should search between in the score image.
        :param min_log_likelihood: The minimum log likelihood allowed for a measurement before it is rejected.
	 */

	int w = score_img.cols, h = score_img.rows;
	if (w != img_w || h != img_h) cerr <<" Window not parametrized for this score_img size" << endl;

	// Apply a column-wise Gaussian filter to score the x-positions in this window's search region
	x_search_range[0] = max(0, int(x_search_range[0]));
    x_search_range[1] = min(int(x_search_range[1]), img_w);

    int x_offset = x_search_range[0];
    Mat search_region = score_img( cv::Range(y_begin, y_end), cv::Range(x_offset, x_search_range[1]));
    Mat col_sum(1,search_region.cols, CV_32FC1, Scalar(0.0));
    Mat column_scores(1,search_region.cols, CV_32FC1, Scalar(0.0));

    // column sum
    cv::reduce(search_region, col_sum, 0, REDUCE_SUM, CV_32FC1);

    // Gaussian filter
    float truncate = 3.0;
    float sd = width / 3.0;
    int ksize = int(truncate * sd + 0.5);
    GaussianBlur(col_sum, column_scores, cv::Size(1,ksize), BORDER_CONSTANT, 0);

    double minVal = 0.0, maxVal = 0.0;
    Point minLoc, maxLoc;
    minMaxLoc(column_scores, &minVal, &maxVal, &minLoc, &maxLoc);
    if (maxVal > 0.0) {
    	detected = true;
    	if (maxVal <= 0.0)
    		x_measured = (x_search_range[0] + x_search_range[1])/2;
    	else
    		x_measured = maxLoc.x + x_offset;


    	int x_begin_measured = x_begin(1);
    	int x_end_measured = x_end(1);

    	float window_magnitude = 0.0; // cv::sum(temp)[0];

    	// verify indexes
    	int x_left_border = max(x_begin_measured, int(x_search_range[0]));
    	int x_right_border = min(x_end_measured, int(x_search_range[1]));

    	for(int i = x_left_border - x_offset; i < x_right_border - x_offset; i++) {
    		if (i < 0 || i >= column_scores.cols)
    			cerr << "index out of range." << endl;
    		window_magnitude += column_scores.at<float>(i);
    	}

    	float noise_magnitude = cv::sum(column_scores)[0] - window_magnitude;
    	float signal_noise_ratio = 0.0;
    	if (window_magnitude > 0)
    		signal_noise_ratio = window_magnitude / (window_magnitude + noise_magnitude);
    	else
    		signal_noise_ratio  = 0.0;

    	if (signal_noise_ratio < 0.6 || filter->loglikelihood(x_measured) < min_log_likelihood) {
    		// in both cases, this measurement is considered as outlier
    		// Suspect / bad measurement, don't update filter/position
    		freeze();
    		return;
    	}

    	unfreeze();
    	filter->update(x_measured);
    	x_filtered = filter->get_position();
    }
    else {
    	detected = false;
    	freeze();
    }

}

}


