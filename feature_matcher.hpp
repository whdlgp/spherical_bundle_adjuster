#pragma once
#include "opencv2/opencv_modules.hpp"

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
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "debug_print.h"

class feature_matcher
{
    public:
    void init();
    void deinit();
    feature_matcher(){ init(); } 
    ~feature_matcher() { deinit(); }

    std::vector<cv::KeyPoint> detect_key_point(const cv::Mat &image);
    cv::Mat comput_descriptor(const cv::Mat &image, std::vector<cv::KeyPoint> &key_point);
    std::vector<cv::DMatch> match_two_image(const cv::Mat &descriptor1, const cv::Mat &descriptor2);
    cv::Mat draw_match(const cv::Mat& im_left, const cv::Mat& im_right, const std::vector<cv::KeyPoint>& key_left, const std::vector<cv::KeyPoint>& key_right);

    void do_all(const cv::Mat &im_left, const cv::Mat &im_right, std::vector<cv::KeyPoint>& left_key, std::vector<cv::KeyPoint>& right_key, int& match_size, cv::Mat& match_output, int& total_key_num);

    private:
    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::Feature2D> descriptor_extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    std::vector<cv::KeyPoint> key_point_left;
    std::vector<cv::KeyPoint> key_point_right;
    cv::Mat descriptor_left;
    cv::Mat descriptor_right;
    std::vector<cv::DMatch> matches;
};
