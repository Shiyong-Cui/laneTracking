//============================================================================
// Name        : lane.cpp
// Author      : Shiyong Cui
// Version     :
// Copyright   : Your copyright notice
// Description : main file for lane tracking
//============================================================================

#include <map>
#include <sstream>
#include <string>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "LaneFinder.h"
#include "Timer.h"
#include <stdlib.h>
#include <stdio.h>

#include <sys/time.h>

using namespace lane;
using namespace std;
using namespace cv;



void test_read_write()
{
	int r = 3, c = 3;
	cv::Mat data(r, c, CV_32FC1, Scalar(0.0));
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c;  j++) {
			data.at<float>(i, j) = rand() % 100;
		}
	}

	// write Mat to file
	cv::FileStorage fs("file.yml", cv::FileStorage::WRITE);
	fs << "yourMat" << data;
	fs.release();

	// read Mat from file
	Mat rdata;
	cv::FileStorage fs2("file.yml", FileStorage::READ);
	fs2["yourMat"] >> rdata;
	cout << rdata << endl;
	fs2.release();
}

int main(int argc, const char** argv) {
	float scale_factor = 0.5;
	bool bcalibration = false;
	vector<cv::Point2f> lane_shape = {Point2f(584, 458), Point2f(701, 458), Point2f(295, 665), Point2f(1022, 665)};
	vector<float> scale_correction = {30 / 720.0, 3.7 / 700};

	 if (bcalibration) {
		int nbfile = 14;
		string pre_fix = "../data/image/";

		vector<string> files(nbfile);
		for(int i = 0; i < nbfile; i++) {
			std::ostringstream stream;
			stream << i+1;
			files[i] = pre_fix + "image-" + stream.str() + ".jpg";
			cout << files[i] << endl;
		}
		// set parameters
		vector<int> chessboard_size = {5, 7};
		DashboardCamera cam(files, chessboard_size, lane_shape, scale_correction);
		cam.saveCameraParameters("cam.yml");
	 }
	 else {
		// calibration file
		string file = "cam.yml";
		// image dimension
		int img_h = 720, img_w = 1280;
		DashboardCamera cam(file, img_h, img_w, lane_shape, scale_correction);

		//open the video file for reading
		VideoCapture cap("../data/test_videos/test_video.mp4");
		// if not success, exit program
		if (cap.isOpened() == false) {
			cout << "Cannot open the video file" << endl;
			cin.get(); //wait for any key press
			return -1;
		}
		// main class for lane fitting
		Timer timer;
		vector<int> window_shape(2,0);
		window_shape[0] = 80; window_shape[1] = 61;
		LaneFinder lane_finder(&cam, window_shape);
		string window_name = "main";
		vector<double> times;
		while (true) {
			Mat frame;
			bool bSuccess = cap.read(frame); // read a new frame from video
			//Breaking the while loop at the end of the video
			if (bSuccess == false) {
				cout << "Found the end of the video" << endl;
				break;
			}

			timer.start();
			Mat res = lane_finder.find_lines(frame);
			timer.stop();
			double t = timer.print("find lane ");
			times.push_back(t);

			if (res.cols > 0)
				imshow(window_name, res);

			//wait for for 10 ms until any key is pressed.
			//If the 'Esc' key is pressed, break the while loop.
			//If the any other key is pressed, continue the loop
			//If any key is not pressed withing 10 ms, continue the loop
			if (waitKey(10) == 27) {
				cout << "Esc key is pressed by user. Stoppig the video" << endl;
				break;
			}
		}
		double average = accumulate( times.begin(), times.end(), 0.0)/times.size();
		cout << "average time is " << average << endl;
	 }



	return 0;
}
