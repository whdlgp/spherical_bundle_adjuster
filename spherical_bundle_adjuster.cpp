#include "spherical_bundle_adjuster.hpp"

using namespace std;
using namespace cv;

using ceres::CENTRAL;
using ceres::CostFunction;
using ceres::NumericDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

void spherical_bundle_adjuster::do_bundle_adjustment(const cv::Mat &im_left, const cv::Mat &im_right)
{
    vector<KeyPoint> left_key;
    vector<KeyPoint> right_key;
    int match_size;
    Mat match_output;

    feature_matcher fm;
    fm.do_all(im_left, im_right, left_key, right_key, match_size, match_output);

    // convert pixel to radian coordinate, in unit sphere
    // x : longitude
    // y : latitude
    double im_width = im_left.cols;
    double im_height = im_left.rows;
    vector<Point2d> key_point_left_radian(match_size);
    vector<Point2d> key_point_right_radian(match_size);
    for(int i = 0; i < match_size; i++)
    {
        key_point_left_radian[i].x = 2*M_PI*(left_key[i].pt.x / im_width);
        key_point_right_radian[i].x = 2*M_PI*(right_key[i].pt.x / im_width);
        key_point_left_radian[i].y = M_PI*(left_key[i].pt.y / im_height);
        key_point_right_radian[i].y = M_PI*(right_key[i].pt.y / im_height);
    }

    // convert radian to rectangular coordinate
    vector<Point3d> key_point_left_rect(match_size);
    vector<Point3d> key_point_right_rect(match_size);
    for(int i = 0; i < match_size; i++)
    {
        key_point_left_rect[i].x = sin(key_point_left_radian[i].y)*cos(key_point_left_radian[i].x);
        key_point_left_rect[i].y = sin(key_point_left_radian[i].y)*sin(key_point_left_radian[i].x);
        key_point_left_rect[i].z = cos(key_point_left_radian[i].y);

        key_point_right_rect[i].x = sin(key_point_right_radian[i].y)*cos(key_point_right_radian[i].x);
        key_point_right_rect[i].y = sin(key_point_right_radian[i].y)*sin(key_point_right_radian[i].x);
        key_point_right_rect[i].z = cos(key_point_right_radian[i].y);
    }

    // Bundle adjustment
    Problem problem;

    // initial value
    vector<array<double, 2>> init_d(match_size);
    for(int i = 0; i < match_size; i++)
    {
        init_d[i][0] = 1;
        init_d[i][1] = 1;
    }
    double init_rot[3] = {expected_roll/180*M_PI, expected_pitch/180*M_PI, expected_yaw/180*M_PI};
    double init_tran[3] = {0.0, 0.0, 0.0};

    // Add residual with initial value
    ba_spherical_costfunctor_rot_only::add_residual(problem, key_point_left_rect, key_point_right_rect, init_rot, init_tran, init_d, match_size);

    // Set options and solve problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = 8;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << "Given thread: " << summary.num_threads_given << std::endl;
    std::cout << "Used thread: " << summary.num_threads_used << std::endl;

    std::cout << "expected rotation vector " << expected_roll << ' ' << expected_pitch << ' ' << expected_yaw << ' ' << std::endl;
    std::cout << "rotation vector in degree " << init_rot[0]/M_PI*180.0 << ' ' << init_rot[1]/M_PI*180.0 << ' ' << init_rot[2]/M_PI*180.0<< std::endl;
    std::cout << "translation vector " << init_tran[0] << ' ' << init_tran[1] << ' ' << init_tran[2] << std::endl;

    ofstream log_file;
    log_file.open("log.txt", std::ios_base::app);
    log_file << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ',' 
             << init_rot[0]/M_PI*180.0 << ',' << init_rot[1]/M_PI*180.0 << ',' << init_rot[2]/M_PI*180.0 << ',' 
             << match_size << endl; 
    
    // Save test match image and log file
    string file_name;
    stringstream file_name_stream(file_name);
    file_name_stream << "match_result/";
    file_name_stream << init_rot[0]/M_PI*180.0 << ',' << init_rot[1]/M_PI*180.0 << ',' << init_rot[2]/M_PI*180.0 << ',' << match_size << ".png" << endl;
    file_name_stream >> file_name;
    cv::imwrite(file_name, match_output);
    
    DEBUG_PRINT_OUT("Done."); 
}

// reprojection error
template <typename T> bool ba_spherical_costfunctor::operator()(const T *const d, const T *const r, const T *const t, T *residual) const
{
    // unit sphere projected point
    T X1[3], X2[3];
    X1[0] = cam1_x_*d[0];
    X1[1] = cam1_y_*d[0];
    X1[2] = cam1_z_*d[0];
    X2[0] = cam2_x_*d[1];
    X2[1] = cam2_y_*d[1];
    X2[2] = cam2_z_*d[1];

    // rotation and translationR(Xn - t)
    T X1_translate[3], X1_RT[3];
    X1_translate[0] = X1[0] - t[0];
    X1_translate[1] = X1[1] - t[1];
    X1_translate[2] = X1[2] - t[2];
    ceres::AngleAxisRotatePoint(r, X1_translate, X1_RT);

    residual[0] = (X2[0] - X1_RT[0])*(X2[0] - X1_RT[0]);
    residual[1] = (X2[1] - X1_RT[1])*(X2[1] - X1_RT[1]);
    residual[2] = (X2[2] - X1_RT[2])*(X2[2] - X1_RT[2]);

    return true;
}

void ba_spherical_costfunctor::add_residual(Problem& problem
                , vector<Point3d>& key_point_left_rect
                , vector<Point3d>& key_point_right_rect
                , double* init_rot
                , double* init_tran
                , vector<array<double, 2>>& init_d
                , int match_num)
{
    for(int i = 0; i < match_num; i++)
    {
        CostFunction *cost_functor = new ceres::AutoDiffCostFunction<ba_spherical_costfunctor, 3, 2, 3, 3>
                                    (new ba_spherical_costfunctor(key_point_left_rect[i].x
                                                                , key_point_left_rect[i].y
                                                                , key_point_left_rect[i].z
                                                                , key_point_right_rect[i].x
                                                                , key_point_right_rect[i].y
                                                                , key_point_right_rect[i].z));
        problem.AddResidualBlock(cost_functor, new ceres::HuberLoss(1.0), init_d[i].data(), init_rot, init_tran);
    }
}

// reprojection error
template <typename T> bool ba_spherical_costfunctor_rot_only::operator()(const T *const r, T *residual) const
{
    // unit sphere projected point
    T X1[3], X2[3];
    T d[2];
    d[0] = static_cast<T>(d_1_);
    d[1] = static_cast<T>(d_2_);
    X1[0] = cam1_x_*d[0];
    X1[1] = cam1_y_*d[0];
    X1[2] = cam1_z_*d[0];
    X2[0] = cam2_x_*d[1];
    X2[1] = cam2_y_*d[1];
    X2[2] = cam2_z_*d[1];

    // rotation and translationR(Xn - t)
    T X1_translate[3], X1_RT[3];
    X1_translate[0] = X1[0] - t_1_;
    X1_translate[1] = X1[1] - t_2_;
    X1_translate[2] = X1[2] - t_3_;
    ceres::AngleAxisRotatePoint(r, X1_translate, X1_RT);

    residual[0] = (X2[0] - X1_RT[0])*(X2[0] - X1_RT[0]);
    residual[1] = (X2[1] - X1_RT[1])*(X2[1] - X1_RT[1]);
    residual[2] = (X2[2] - X1_RT[2])*(X2[2] - X1_RT[2]);

    return true;
}

void ba_spherical_costfunctor_rot_only::add_residual(Problem& problem
                         , vector<Point3d>& key_point_left_rect
                         , vector<Point3d>& key_point_right_rect
                         , double* init_rot
                         , double* init_tran
                         , vector<array<double, 2>>& init_d
                         , int match_num)
{
    for(int i = 0; i < match_num; i++)
    {
        CostFunction *cost_functor = new ceres::AutoDiffCostFunction<ba_spherical_costfunctor_rot_only, 3, 3>
                                    (new ba_spherical_costfunctor_rot_only(key_point_left_rect[i].x
                                                                , key_point_left_rect[i].y
                                                                , key_point_left_rect[i].z
                                                                , key_point_right_rect[i].x
                                                                , key_point_right_rect[i].y
                                                                , key_point_right_rect[i].z
                                                                , init_tran[0]
                                                                , init_tran[1]
                                                                , init_tran[2]
                                                                , init_d[0][0]
                                                                , init_d[1][0]));
        problem.AddResidualBlock(cost_functor, new ceres::HuberLoss(1.0), init_rot);
    }
}
