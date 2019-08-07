#pragma once
#define _USE_MATH_DEFINES
#include <vector>
#include <algorithm>
#include <utility>
#include <set>
#include <functional>
#include <sstream>
#include <iostream>
#include <cmath>

#include "opencv2/core/ocl.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

#include <omp.h>

class equi2cube
{
    public:
    void set_omp(int num_proc);
    cv::Mat get_back(const cv::Mat& im, int cube_size);
    cv::Mat get_front(const cv::Mat& im, int cube_size);
    cv::Mat get_left(const cv::Mat& im, int cube_size);
    cv::Mat get_right(const cv::Mat& im, int cube_size);
    cv::Mat get_top(const cv::Mat& im, int cube_size);
    cv::Mat get_bottom(const cv::Mat& im, int cube_size);
    cv::Mat get_all(const cv::Mat& im, int cube_size);
    private:
};