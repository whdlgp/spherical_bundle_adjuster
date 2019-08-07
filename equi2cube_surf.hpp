#pragma once

#include "feature_matcher.hpp"
#include "equi2cube.hpp"
#include <omp.h>

class equi2cube_surf
{
    public:
    void set_omp(int num_proc);
    void set_cube_size(int cube_size);
    void cube2equi_pixel(cv::Point2f& cube_pixel, cv::Point2f& equi_pixel, int cube_size, int im_width, int im_height);
    void do_all(const cv::Mat &im_left, const cv::Mat &im_right, std::vector<cv::KeyPoint>& left_key, std::vector<cv::KeyPoint>& right_key, int& match_size, cv::Mat& match_output, int& total_key_num);

    private:
    int num_proc;
    int cube_size;
};