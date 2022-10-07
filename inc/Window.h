/*
 * Window.h
 *
 *  Created on: Jan 24, 2018
 *      Author: cui
 */

#ifndef WINDOW_H_
#define WINDOW_H_

#include <vector>
#include "WindowFilter.h"

using namespace std;

namespace lane {


class Window {
public:
	Window(int level, vector<int> window_shape, vector<int> img_shape, float x_init, int max_frozen_dur);
	virtual ~Window();

	int x_begin(int opt = 0);
	int x_end(int opt = 0);
	vector<int> pos_xy(int opt = 0);
	void pos_xy(int& x, int& y, int opt = 0);
	float area() { return width * height; };
	void freeze();
	void unfreeze();
	bool droped() { return frozen_dur > max_frozen_dur; }
	Mat get_mask(int opt = 0);
	void update(const Mat& score_img, vector<float>& x_search_range, float min_log_likelihood=-40);

	float getY() const { return y; }
	void setY(float y) { this->y = y;}

	bool isFrozen() const {
		return frozen;
	}

	bool isDetected() const {
		return detected;
	}

	float getFiltered() const {
		return x_filtered;
	}

	int getWidth() const {
		return width;
	}

	int getImgH() const {
		return img_h;
	}

	int getImgW() const {
		return img_w;
	}

	int getYBegin() const {
		return y_begin;
	}

	int getYEnd() const {
		return y_end;
	}

private:
	// Image info
	int img_h;
	int img_w;

	// window shape
	int height;
	int width;
	int y_begin;
	int y_end;

	// Window position
	float x_filtered;
	float y;
	int level;

	// Detection info
	WindowFilter* filter;
	float x_measured;
	bool detected;
	int max_frozen_dur;
	bool frozen;
	int frozen_dur;
	int undrop_buffer;




};

}
#endif /* WINDOW_H_ */
