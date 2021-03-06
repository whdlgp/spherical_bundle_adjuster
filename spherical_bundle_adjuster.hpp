#pragma once

#include "spherical_surf.hpp"

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "debug_print.h"

#include <sstream>
#include <fstream>

#include <omp.h>

class spherical_bundle_adjuster
{
    public:
    spherical_bundle_adjuster(double roll = 0, double pitch = 0, double yaw = 0, double tx = 0, double ty = 0, double tz = 0, double d = 0)
    : expected_roll(roll), expected_pitch(pitch), expected_yaw(yaw), expected_tx(tx), expected_ty(ty), expected_tz(tz), expected_d(d) {} 
    ~spherical_bundle_adjuster() {}
    
    void set_omp(int num_proc);
    void do_bundle_adjustment(const cv::Mat &im_left, const cv::Mat &im_right);
    
    private:
    double max_vec(cv::Vec3f& vec);
    void eight_point_estimation(int im_width, int im_height
                                , std::vector<cv::Point3d>& key_point_left_rect, std::vector<cv::Point3d>& key_point_right_rect
                                , cv::Vec3f& R1_vec, cv::Vec3f& R2_vec, cv::Vec3f& T_vec
                                , bool& R1_valid, bool& R2_valid
                                , int match_size);
    void initial_guess(int im_width, int im_height
                        , std::vector<cv::Point3d>& key_point_left_rect, std::vector<cv::Point3d>& key_point_right_rect
                        , cv::Vec3f& R_vec_out, cv::Vec3f& T_vec_out
                        , int match_size);

    void solve_problem(ceres::Solver::Options& opt
                    , std::vector<cv::Point3d>& key_point_left_rect
                    , std::vector<cv::Point3d>& key_point_right_rect
                    , double* init_rot
                    , double* init_tran
                    , std::vector<std::array<double, 2>>& init_d
                    , int match_num);
    
    void write_log_d(std::vector<std::array<double, 2UL>>& init_d, cv::String name);
    void write_d_circle(const cv::Mat& im_left, std::vector<std::array<double, 2UL>>& init_d, std::vector<cv::KeyPoint> left_key, cv::String name);

    double expected_roll;
    double expected_pitch;
    double expected_yaw;
    double expected_tx;
    double expected_ty;
    double expected_tz;
    double expected_d;

    int num_proc;
};

// spherical bundle adjustment
struct ba_spherical_costfunctor
{
    ba_spherical_costfunctor(double cam1_x, double cam1_y, double cam1_z, double cam2_x, double cam2_y, double cam2_z)
    : cam1_x_(cam1_x), cam1_y_(cam1_y), cam1_z_(cam1_z), cam2_x_(cam2_x), cam2_y_(cam2_y), cam2_z_(cam2_z)
    {}

    // reprojection error
    template <typename T> bool operator()(const T *const d, const T *const r, const T *const t, T *residual) const;

    static void add_residual(ceres::Problem& problem
                           , std::vector<cv::Point3d>& key_point_left_rect
                           , std::vector<cv::Point3d>& key_point_right_rect
                           , double* init_rot
                           , double* init_tran
                           , std::vector<std::array<double, 2>>& init_d
                           , int match_num);

    private:
    const double cam1_x_;
    const double cam1_y_;
    const double cam1_z_;
    const double cam2_x_;
    const double cam2_y_;
    const double cam2_z_;
};

struct ba_spherical_costfunctor_rot_only
{
    ba_spherical_costfunctor_rot_only(double cam1_x, double cam1_y, double cam1_z, double cam2_x, double cam2_y, double cam2_z, double t_1, double t_2, double t_3, double d_1, double d_2)
    : cam1_x_(cam1_x), cam1_y_(cam1_y), cam1_z_(cam1_z), cam2_x_(cam2_x), cam2_y_(cam2_y), cam2_z_(cam2_z), t_1_(t_1), t_2_(t_2), t_3_(t_3), d_1_(d_1), d_2_(d_2)
    {}

    // reprojection error
    template <typename T> bool operator()(const T *const r, T *residual) const;

    static void add_residual(ceres::Problem& problem
                           , std::vector<cv::Point3d>& key_point_left_rect
                           , std::vector<cv::Point3d>& key_point_right_rect
                           , double* init_rot
                           , double* init_tran
                           , std::vector<std::array<double, 2>>& init_d
                           , int match_num);

    private:
    const double cam1_x_;
    const double cam1_y_;
    const double cam1_z_;
    const double cam2_x_;
    const double cam2_y_;
    const double cam2_z_;
    const double t_1_;
    const double t_2_;
    const double t_3_;
    const double d_1_;
    const double d_2_;
};

struct ba_spherical_costfunctor_tran_only
{
    ba_spherical_costfunctor_tran_only(double cam1_x, double cam1_y, double cam1_z, double cam2_x, double cam2_y, double cam2_z, double r_1, double r_2, double r_3, double d_1, double d_2)
    : cam1_x_(cam1_x), cam1_y_(cam1_y), cam1_z_(cam1_z), cam2_x_(cam2_x), cam2_y_(cam2_y), cam2_z_(cam2_z), r_1_(r_1), r_2_(r_2), r_3_(r_3), d_1_(d_1), d_2_(d_2)
    {}

    // reprojection error
    template <typename T> bool operator()(const T *const t, T *residual) const;

    static void add_residual(ceres::Problem& problem
                           , std::vector<cv::Point3d>& key_point_left_rect
                           , std::vector<cv::Point3d>& key_point_right_rect
                           , double* init_rot
                           , double* init_tran
                           , std::vector<std::array<double, 2>>& init_d
                           , int match_num);

    private:
    const double cam1_x_;
    const double cam1_y_;
    const double cam1_z_;
    const double cam2_x_;
    const double cam2_y_;
    const double cam2_z_;
    const double r_1_;
    const double r_2_;
    const double r_3_;
    const double d_1_;
    const double d_2_;
};

struct ba_spherical_costfunctor_d_only
{
    ba_spherical_costfunctor_d_only(double cam1_x, double cam1_y, double cam1_z, double cam2_x, double cam2_y, double cam2_z, double t_1, double t_2, double t_3, double r_1, double r_2, double r_3, double lambda, double c)
    : cam1_x_(cam1_x), cam1_y_(cam1_y), cam1_z_(cam1_z), cam2_x_(cam2_x), cam2_y_(cam2_y), cam2_z_(cam2_z), t_1_(t_1), t_2_(t_2), t_3_(t_3), r_1_(r_1), r_2_(r_2), r_3_(r_3), lambda_(lambda), c_(c)
    {}

    // reprojection error
    template <typename T> bool operator()(const T *const t, T *residual) const;

    static void add_residual(ceres::Problem& problem
                           , std::vector<cv::Point3d>& key_point_left_rect
                           , std::vector<cv::Point3d>& key_point_right_rect
                           , double* init_rot
                           , double* init_tran
                           , std::vector<std::array<double, 2>>& init_d
                           , int match_num);

    private:
    const double cam1_x_;
    const double cam1_y_;
    const double cam1_z_;
    const double cam2_x_;
    const double cam2_y_;
    const double cam2_z_;
    const double t_1_;
    const double t_2_;
    const double t_3_;
    const double r_1_;
    const double r_2_;
    const double r_3_;
    const double lambda_;
    const double c_;
};

class random_array
{
    public:
    random_array(int size)
    : rand_arr(size)
    {
        size_ = size;
        count_ = 0;
        rand_idx_generate();
    }

    int get_rand()
    {
        int retval = rand_arr[count_];
        count_++;
        count_ = count_ % size_;
        return retval;
    }

    private:
    int size_;
    std::vector<int> rand_arr;
    int count_;

    void rand_idx_generate()
    {
        std::iota(rand_arr.begin(), rand_arr.end(), 0);
        std::random_shuffle(rand_arr.begin(), rand_arr.end());
    }
};