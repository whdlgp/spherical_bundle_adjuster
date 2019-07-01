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

void spherical_bundle_adjuster::solve_problem(ceres::Solver::Options& opt
                                            , std::vector<cv::Point3d>& key_point_left_rect
                                            , std::vector<cv::Point3d>& key_point_right_rect
                                            , double* init_rot
                                            , double* init_tran
                                            , std::vector<std::array<double, 2>>& init_d
                                            , int match_num)
{
    Problem problem_rot;
    Problem problem_tran;
    Problem problem_d;
    Solver::Summary summary;
    
    ba_spherical_costfunctor_rot_only::add_residual(problem_rot, key_point_left_rect, key_point_right_rect, init_rot, init_tran, init_d, match_num);
    Solve(opt, &problem_rot, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Given thread: " << summary.num_threads_given << std::endl;
    std::cout << "Used thread: " << summary.num_threads_used << std::endl;

    ba_spherical_costfunctor_tran_only::add_residual(problem_tran, key_point_left_rect, key_point_right_rect, init_rot, init_tran, init_d, match_num);
    Solve(opt, &problem_tran, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Given thread: " << summary.num_threads_given << std::endl;
    std::cout << "Used thread: " << summary.num_threads_used << std::endl;

    ba_spherical_costfunctor_d_only::add_residual(problem_d, key_point_left_rect, key_point_right_rect, init_rot, init_tran, init_d, match_num);
    Solve(opt, &problem_d, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Given thread: " << summary.num_threads_given << std::endl;
    std::cout << "Used thread: " << summary.num_threads_used << std::endl;

    std::cout << "expected rotation vector " << expected_roll << ' ' << expected_pitch << ' ' << expected_yaw << ' ' << std::endl;
    std::cout << "rotation vector in degree " << init_rot[0]/M_PI*180.0 << ' ' << init_rot[1]/M_PI*180.0 << ' ' << init_rot[2]/M_PI*180.0<< std::endl;
    std::cout << "translation vector " << init_tran[0] << ' ' << init_tran[1] << ' ' << init_tran[2] << std::endl;
}

void spherical_bundle_adjuster::write_log_d(std::vector<std::array<double, 2UL>>& init_d, cv::String name)
{
    ofstream log_d_file;
    log_d_file.open(name + ".txt", std::ios_base::app);
    for(int i = 0; i < init_d.size(); i++)
        log_d_file << init_d[i][0] << ',' << init_d[i][1] << endl;
}

void spherical_bundle_adjuster::write_d_circle(const cv::Mat& im_left, std::vector<std::array<double, 2UL>>& init_d, std::vector<cv::KeyPoint> left_key, cv::String name)
{
    int match_size = init_d.size();

    double max_d = 0, min_d = 0;
    for(int i = 0; i < match_size; i++)
    {
        if(init_d[i][0] > max_d) max_d = init_d[i][0];
        if(init_d[i][0] < min_d) min_d = init_d[i][0];
    }
    cout << max_d << ',' << min_d << endl;

    Mat im_left_d = im_left.clone();
    for(int i = 0; i < match_size; i++)
    {
        if(init_d[i][0] >= 0)
            circle(im_left_d, left_key[i].pt, 10, Scalar(0, init_d[i][0]*255/max_d, 0), 5);
        else
            circle(im_left_d, left_key[i].pt, 10, Scalar(0, 0, 255-init_d[i][0]*255/min_d), 5);
    }
    string d_name;
    stringstream d_name_stream(d_name);
    d_name_stream << "match_result/";
    d_name_stream << name << ".png" << endl;
    d_name_stream >> d_name;
    cv::imwrite(d_name, im_left_d);
}

void spherical_bundle_adjuster::do_bundle_adjustment(const cv::Mat &im_left, const cv::Mat &im_right)
{
    vector<KeyPoint> left_key;
    vector<KeyPoint> right_key;
    int match_size;
    Mat match_output;
    int total_key_num;

    DEBUG_PRINT_OUT("Do feature finding and matching");
    spherical_surf fm;
    fm.set_omp(this->num_proc);
    fm.do_all(im_left, im_right, left_key, right_key, match_size, match_output, total_key_num);

    // convert pixel to radian coordinate, in unit sphere
    // x : longitude
    // y : latitude
    double im_width = im_left.cols;
    double im_height = im_left.rows;
    vector<Point2d> key_point_left_radian(match_size);
    vector<Point2d> key_point_right_radian(match_size);

    #pragma omp parallel for
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
    #pragma omp parallel for
    for(int i = 0; i < match_size; i++)
    {
        key_point_left_rect[i].x = sin(key_point_left_radian[i].y)*cos(key_point_left_radian[i].x);
        key_point_left_rect[i].y = sin(key_point_left_radian[i].y)*sin(key_point_left_radian[i].x);
        key_point_left_rect[i].z = cos(key_point_left_radian[i].y);

        key_point_right_rect[i].x = sin(key_point_right_radian[i].y)*cos(key_point_right_radian[i].x);
        key_point_right_rect[i].y = sin(key_point_right_radian[i].y)*sin(key_point_right_radian[i].x);
        key_point_right_rect[i].z = cos(key_point_right_radian[i].y);
    }

    DEBUG_PRINT_OUT("Do bundle adjustment");

    // initial value
    vector<array<double, 2>> init_d(match_size);
    for(int i = 0; i < match_size; i++)
    {
        init_d[i][0] = 1;
        init_d[i][1] = 1;
    }
    double init_rot[3] = {expected_roll/180*M_PI, expected_pitch/180*M_PI, expected_yaw/180*M_PI};
    double init_tran[3] = {0.0, 0.0, 0.0};
    
    // Set options and solve problem
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = true;
    options.num_threads = this->num_proc;

    solve_problem(options
                , key_point_left_rect
                , key_point_right_rect
                , init_rot
                , init_tran
                , init_d
                , match_size);

    ofstream log_file;
    log_file.open("log.txt", std::ios_base::app);
    log_file << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ',' 
             << init_rot[0]/M_PI*180.0 << ',' << init_rot[1]/M_PI*180.0 << ',' << init_rot[2]/M_PI*180.0 << ',' 
             << init_tran[0] << ',' << init_tran[0] << ',' << init_tran[0] << ','
             << match_size << endl; 
    log_file.close();

    write_d_circle(im_left, init_d, left_key, "d_found");
    write_log_d(init_d, "log_d");
    
    {// second initial value
        vector<KeyPoint> left_key2;
        vector<KeyPoint> right_key2;
        vector<Point3d> key_point_left_rect2;
        vector<Point3d> key_point_right_rect2;
        vector<array<double, 2>> init_d2;
        int match_size2 = 0;
        for(int i = 0; i < match_size; i++)
        {
            if(init_d[i][0] >= 0)
            {
                match_size2++;
                left_key2.push_back(left_key[i]);
                right_key2.push_back(right_key[i]);
                key_point_left_rect2.push_back(key_point_left_rect[i]);
                key_point_right_rect2.push_back(key_point_right_rect[i]);
                array<double, 2> tmp = {1, 1};
                init_d2.push_back(tmp);
            }
        }
        double init_rot2[3] = {expected_roll/180*M_PI, expected_pitch/180*M_PI, expected_yaw/180*M_PI};
        double init_tran2[3] = {0.0, 0.0, 0.0};

        solve_problem(options
                    , key_point_left_rect2
                    , key_point_right_rect2
                    , init_rot2
                    , init_tran2
                    , init_d2
                    , match_size2);
                    
        ofstream log_file;
        log_file.open("log.txt", std::ios_base::app);
        log_file << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ',' 
                << init_rot2[0]/M_PI*180.0 << ',' << init_rot2[1]/M_PI*180.0 << ',' << init_rot2[2]/M_PI*180.0 << ',' 
                << init_tran2[0] << ',' << init_tran2[0] << ',' << init_tran2[0] << ','
                << match_size << endl; 
        log_file.close();
        
        write_d_circle(im_left, init_d2, left_key2, "d_found2");
        write_log_d(init_d2, "log_d2");
    }
    
    {// Third initial value
        vector<KeyPoint> left_key3;
        vector<KeyPoint> right_key3;
        vector<Point3d> key_point_left_rect3;
        vector<Point3d> key_point_right_rect3;
        vector<array<double, 2>> init_d3;
        int match_size3 = 0;
        for(int i = 0; i < match_size; i++)
        {
            if(init_d[i][0] >= 0)
            {
                match_size3++;
                left_key3.push_back(left_key[i]);
                right_key3.push_back(right_key[i]);
                key_point_left_rect3.push_back(key_point_left_rect[i]);
                key_point_right_rect3.push_back(key_point_right_rect[i]);
                init_d3.push_back(init_d[i]);
            }
        }
        double init_rot3[3] = {init_rot[0], init_rot[1], init_rot[2]};
        double init_tran3[3] = {init_tran[0], init_tran[1], init_tran[2]};

        solve_problem(options
                    , key_point_left_rect3
                    , key_point_right_rect3
                    , init_rot3
                    , init_tran3
                    , init_d3
                    , match_size3);

        ofstream log_file;
        log_file.open("log.txt", std::ios_base::app);
        log_file << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ',' 
                << init_rot3[0]/M_PI*180.0 << ',' << init_rot3[1]/M_PI*180.0 << ',' << init_rot3[2]/M_PI*180.0 << ',' 
                << init_tran3[0] << ',' << init_tran3[0] << ',' << init_tran3[0] << ','
                << match_size << endl; 
        log_file.close();

        write_d_circle(im_left, init_d3, left_key3, "d_found3");
        write_log_d(init_d3, "log_d3");
    }
    
    {// 4th initial value
        vector<KeyPoint> left_key4;
        vector<KeyPoint> right_key4;
        vector<Point3d> key_point_left_rect4;
        vector<Point3d> key_point_right_rect4;
        vector<array<double, 2>> init_d4;
        int match_size4 = 0;
        for(int i = 0; i < match_size; i++)
        {
            if(init_d[i][0] >= 0)
            {
                match_size4++;
                left_key4.push_back(left_key[i]);
                right_key4.push_back(right_key[i]);
                key_point_left_rect4.push_back(key_point_left_rect[i]);
                key_point_right_rect4.push_back(key_point_right_rect[i]);
                init_d4.push_back(init_d[i]);
            }
        }
        double init_rot4[3] = {init_rot[0], init_rot[1], init_rot[2]};
        double init_tran4[3] = {init_tran[0], init_tran[1], init_tran[2]};

        for(int i = 0; i < 2; i++)
        {
            solve_problem(options
                        , key_point_left_rect4
                        , key_point_right_rect4
                        , init_rot4
                        , init_tran4
                        , init_d4
                        , match_size4);
        }

        ofstream log_file;
        log_file.open("log.txt", std::ios_base::app);
        log_file << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ',' 
                << init_rot4[0]/M_PI*180.0 << ',' << init_rot4[1]/M_PI*180.0 << ',' << init_rot4[2]/M_PI*180.0 << ',' 
                << init_tran4[0] << ',' << init_tran4[0] << ',' << init_tran4[0] << ','
                << match_size << endl; 
        log_file.close();

        write_d_circle(im_left, init_d4, left_key4, "d_found4");
        write_log_d(init_d4, "log_d4");
    }
    
    {// 5th initial value
        vector<KeyPoint> left_key5;
        vector<KeyPoint> right_key5;
        vector<Point3d> key_point_left_rect5;
        vector<Point3d> key_point_right_rect5;
        vector<array<double, 2>> init_d5;
        int match_size5 = 0;
        for(int i = 0; i < match_size; i++)
        {
            if(init_d[i][0] >= 0)
            {
                match_size5++;
                left_key5.push_back(left_key[i]);
                right_key5.push_back(right_key[i]);
                key_point_left_rect5.push_back(key_point_left_rect[i]);
                key_point_right_rect5.push_back(key_point_right_rect[i]);
                init_d5.push_back(init_d[i]);
            }
        }
        double init_rot5[3] = {init_rot[0], init_rot[1], init_rot[2]};
        double init_tran5[3] = {init_tran[0], init_tran[1], init_tran[2]};

        for(int i = 0; i < 2; i++)
        {
            for(int i = 0; i < match_size5; i++)
            {
                if(init_d5[i][0] < 0)
                {
                    match_size5--;
                    left_key5.erase(left_key5.begin()+i);
                    right_key5.erase(right_key5.begin()+i);
                    key_point_left_rect5.erase(key_point_left_rect5.begin()+i);
                    key_point_right_rect5.erase(key_point_right_rect5.begin()+i);
                    init_d5.erase(init_d5.begin()+i);
                }
            }

            solve_problem(options
                        , key_point_left_rect5
                        , key_point_right_rect5
                        , init_rot5
                        , init_tran5
                        , init_d5
                        , match_size5);
        }

        ofstream log_file;
        log_file.open("log.txt", std::ios_base::app);
        log_file << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ',' 
                << init_rot5[0]/M_PI*180.0 << ',' << init_rot5[1]/M_PI*180.0 << ',' << init_rot5[2]/M_PI*180.0 << ',' 
                << init_tran5[0] << ',' << init_tran5[0] << ',' << init_tran5[0] << ','
                << match_size << endl; 
        log_file.close();

        write_d_circle(im_left, init_d5, left_key5, "d_found5");
        write_log_d(init_d5, "log_d5");
    }

    {// 6th initial value
        vector<KeyPoint> left_key6;
        vector<KeyPoint> right_key6;
        vector<Point3d> key_point_left_rect6;
        vector<Point3d> key_point_right_rect6;
        vector<array<double, 2>> init_d6;
        int match_size6 = 0;
        for(int i = 0; i < match_size; i++)
        {
            if(init_d[i][0] >= 0)
            {
                match_size6++;
                left_key6.push_back(left_key[i]);
                right_key6.push_back(right_key[i]);
                key_point_left_rect6.push_back(key_point_left_rect[i]);
                key_point_right_rect6.push_back(key_point_right_rect[i]);
                init_d6.push_back(init_d[i]);
            }
        }
        double init_rot6[3] = {init_rot[0], init_rot[1], init_rot[2]};
        double init_tran6[3] = {init_tran[0], init_tran[1], init_tran[2]};

        for(int i = 0; i < 2; i++)
        {
            for(int i = 0; i < match_size6; i++)
            {
                if(init_d6[i][0] < 0)
                {
                    match_size6--;
                    left_key6.erase(left_key6.begin()+i);
                    right_key6.erase(right_key6.begin()+i);
                    key_point_left_rect6.erase(key_point_left_rect6.begin()+i);
                    key_point_right_rect6.erase(key_point_right_rect6.begin()+i);
                    init_d6.erase(init_d6.begin()+i);
                }
            }

            double max_d = 0, min_d = 0;
            int max_idx = 0, min_idx = 0;
            for(int i = 0; i < match_size6; i++)
            {
                if(init_d6[i][0] > max_d) 
                {
                    max_d = init_d6[i][0];
                    max_idx = i;
                }
                if(init_d6[i][0] < min_d)
                {
                    min_d = init_d6[i][0];
                    min_idx = i;
                }
            }
            match_size6--;
            left_key6.erase(left_key6.begin()+max_idx);
            right_key6.erase(right_key6.begin()+max_idx);
            key_point_left_rect6.erase(key_point_left_rect6.begin()+max_idx);
            key_point_right_rect6.erase(key_point_right_rect6.begin()+max_idx);
            init_d6.erase(init_d6.begin()+max_idx);

            match_size6--;
            left_key6.erase(left_key6.begin()+min_idx);
            right_key6.erase(right_key6.begin()+min_idx);
            key_point_left_rect6.erase(key_point_left_rect6.begin()+min_idx);
            key_point_right_rect6.erase(key_point_right_rect6.begin()+min_idx);
            init_d6.erase(init_d6.begin()+min_idx);
            
            solve_problem(options
                        , key_point_left_rect6
                        , key_point_right_rect6
                        , init_rot6
                        , init_tran6
                        , init_d6
                        , match_size6);
        }

        ofstream log_file;
        log_file.open("log.txt", std::ios_base::app);
        log_file << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ',' 
                << init_rot6[0]/M_PI*180.0 << ',' << init_rot6[1]/M_PI*180.0 << ',' << init_rot6[2]/M_PI*180.0 << ',' 
                << init_tran6[0] << ',' << init_tran6[0] << ',' << init_tran6[0] << ','
                << match_size << endl; 
        log_file.close();

        write_d_circle(im_left, init_d6, left_key6, "d_found6");
        write_log_d(init_d6, "log_d6");
    }
    
    {// 7th initial value
        vector<KeyPoint> left_key7;
        vector<KeyPoint> right_key7;
        vector<Point3d> key_point_left_rect7;
        vector<Point3d> key_point_right_rect7;
        vector<array<double, 2>> init_d7;
        int match_size7 = 0;
        for(int i = 0; i < match_size; i++)
        {
            if(init_d[i][0] >= 0)
            {
                match_size7++;
                left_key7.push_back(left_key[i]);
                right_key7.push_back(right_key[i]);
                key_point_left_rect7.push_back(key_point_left_rect[i]);
                key_point_right_rect7.push_back(key_point_right_rect[i]);
                array<double, 2> tmp = {1, 1};
                init_d7.push_back(tmp);
            }
        }
        double init_rot7[3] = {expected_roll/180*M_PI, expected_pitch/180*M_PI, expected_yaw/180*M_PI};
        double init_tran7[3] = {0.0, 0.0, 0.0};

        for(int i = 0; i < 2; i++)
        {
            for(int i = 0; i < match_size7; i++)
            {
                if(init_d7[i][0] < 0)
                {
                    match_size7--;
                    left_key7.erase(left_key7.begin()+i);
                    right_key7.erase(right_key7.begin()+i);
                    key_point_left_rect7.erase(key_point_left_rect7.begin()+i);
                    key_point_right_rect7.erase(key_point_right_rect7.begin()+i);
                    init_d7.erase(init_d7.begin()+i);
                }
            }

            double max_d = 0, min_d = 0;
            int max_idx = 0, min_idx = 0;
            for(int i = 0; i < match_size7; i++)
            {
                if(init_d7[i][0] > max_d) 
                {
                    max_d = init_d7[i][0];
                    max_idx = i;
                }
                if(init_d7[i][0] < min_d)
                {
                    min_d = init_d7[i][0];
                    min_idx = i;
                }
            }
            match_size7--;
            left_key7.erase(left_key7.begin()+max_idx);
            right_key7.erase(right_key7.begin()+max_idx);
            key_point_left_rect7.erase(key_point_left_rect7.begin()+max_idx);
            key_point_right_rect7.erase(key_point_right_rect7.begin()+max_idx);
            init_d7.erase(init_d7.begin()+max_idx);

            match_size7--;
            left_key7.erase(left_key7.begin()+min_idx);
            right_key7.erase(right_key7.begin()+min_idx);
            key_point_left_rect7.erase(key_point_left_rect7.begin()+min_idx);
            key_point_right_rect7.erase(key_point_right_rect7.begin()+min_idx);
            init_d7.erase(init_d7.begin()+min_idx);
            
            solve_problem(options
                        , key_point_left_rect7
                        , key_point_right_rect7
                        , init_rot7
                        , init_tran7
                        , init_d7
                        , match_size7);
        }

        ofstream log_file;
        log_file.open("log.txt", std::ios_base::app);
        log_file << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ',' 
                << init_rot7[0]/M_PI*180.0 << ',' << init_rot7[1]/M_PI*180.0 << ',' << init_rot7[2]/M_PI*180.0 << ',' 
                << init_tran7[0] << ',' << init_tran7[0] << ',' << init_tran7[0] << ','
                << match_size << endl; 
        log_file.close();

        write_d_circle(im_left, init_d7, left_key7, "d_found7");
        write_log_d(init_d7, "log_d7");
    }

    {// 8th initial value
        vector<KeyPoint> left_key8;
        vector<KeyPoint> right_key8;
        vector<Point3d> key_point_left_rect8;
        vector<Point3d> key_point_right_rect8;
        vector<array<double, 2>> init_d8;
        int match_size8 = 0;
        for(int i = 0; i < match_size; i++)
        {
            if(init_d[i][0] >= 0)
            {
                match_size8++;
                left_key8.push_back(left_key[i]);
                right_key8.push_back(right_key[i]);
                key_point_left_rect8.push_back(key_point_left_rect[i]);
                key_point_right_rect8.push_back(key_point_right_rect[i]);
                array<double, 2> tmp = {1, 1};
                init_d8.push_back(tmp);
            }
        }
        double init_rot8[3] = {expected_roll/180*M_PI, expected_pitch/180*M_PI, expected_yaw/180*M_PI};
        double init_tran8[3] = {0.0, 0.0, 0.0};

        for(int i = 0; i < 2; i++)
        {
            for(int i = 0; i < match_size8; i++)
            {
                if(init_d8[i][0] < 0)
                {
                    match_size8--;
                    left_key8.erase(left_key8.begin()+i);
                    right_key8.erase(right_key8.begin()+i);
                    key_point_left_rect8.erase(key_point_left_rect8.begin()+i);
                    key_point_right_rect8.erase(key_point_right_rect8.begin()+i);
                    init_d8.erase(init_d8.begin()+i);
                }
            }

            double max_d = 0, min_d = 0;
            int max_idx = 0, min_idx = 0;
            for(int i = 0; i < match_size8; i++)
            {
                if(init_d8[i][0] > max_d) 
                {
                    max_d = init_d8[i][0];
                    max_idx = i;
                }
                if(init_d8[i][0] < min_d)
                {
                    min_d = init_d8[i][0];
                    min_idx = i;
                }
            }
            match_size8--;
            left_key8.erase(left_key8.begin()+max_idx);
            right_key8.erase(right_key8.begin()+max_idx);
            key_point_left_rect8.erase(key_point_left_rect8.begin()+max_idx);
            key_point_right_rect8.erase(key_point_right_rect8.begin()+max_idx);
            init_d8.erase(init_d8.begin()+max_idx);

            match_size8--;
            left_key8.erase(left_key8.begin()+min_idx);
            right_key8.erase(right_key8.begin()+min_idx);
            key_point_left_rect8.erase(key_point_left_rect8.begin()+min_idx);
            key_point_right_rect8.erase(key_point_right_rect8.begin()+min_idx);
            init_d8.erase(init_d8.begin()+min_idx);

            init_rot8[0] = expected_roll/180*M_PI;
            init_rot8[1] = expected_pitch/180*M_PI;
            init_rot8[2] = expected_yaw/180*M_PI;
            init_tran8[0] = 0.0;
            init_tran8[1] = 0.0;
            init_tran8[2] = 0.0;
            init_d8.clear();
            for(int i = 0; i < match_size8; i++)
            {
                array<double, 2> tmp = {1, 1};
                init_d8.push_back(tmp);
            }
            
            solve_problem(options
                        , key_point_left_rect8
                        , key_point_right_rect8
                        , init_rot8
                        , init_tran8
                        , init_d8
                        , match_size8);
        }

        ofstream log_file;
        log_file.open("log.txt", std::ios_base::app);
        log_file << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ',' 
                << init_rot8[0]/M_PI*180.0 << ',' << init_rot8[1]/M_PI*180.0 << ',' << init_rot8[2]/M_PI*180.0 << ',' 
                << init_tran8[0] << ',' << init_tran8[0] << ',' << init_tran8[0] << ','
                << match_size << endl; 
        log_file.close();

        write_d_circle(im_left, init_d8, left_key8, "d_found8");
        write_log_d(init_d8, "log_d8");
    }


    // Save test match image and log file
    string file_name;
    stringstream file_name_stream(file_name);
    file_name_stream << "match_result/";
    file_name_stream << init_rot[0]/M_PI*180.0 << ',' << init_rot[1]/M_PI*180.0 << ',' << init_rot[2]/M_PI*180.0 << ',' << match_size << ".png" << endl;
    file_name_stream >> file_name;
    cv::imwrite(file_name, match_output);
    
    DEBUG_PRINT_OUT("Done."); 
}

void spherical_bundle_adjuster::set_omp(int num_proc)
{
    this->num_proc = num_proc;

    omp_set_num_threads(this->num_proc);
    DEBUG_PRINT_OUT("Number of process: " << this->num_proc);
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

    // rotation and translation (R*Xn - t)
    T X1_rotated[3], X1_RT[3];
    ceres::AngleAxisRotatePoint(r, X1, X1_rotated);

    X1_RT[0] = X1_rotated[0] - t[0];
    X1_RT[1] = X1_rotated[1] - t[1];
    X1_RT[2] = X1_rotated[2] - t[2];

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

    // rotation and translation (R*Xn - t)
    T X1_rotated[3], X1_RT[3];
    ceres::AngleAxisRotatePoint(r, X1, X1_rotated);

    X1_RT[0] = X1_rotated[0] - t_1_;
    X1_RT[1] = X1_rotated[1] - t_2_;
    X1_RT[2] = X1_rotated[2] - t_3_;

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

// reprojection error
template <typename T> bool ba_spherical_costfunctor_tran_only::operator()(const T *const t, T *residual) const
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

    // rotation and translation (R*Xn - t)
    T X1_rotated[3], X1_RT[3];
    T r_vec[3] = {static_cast<T>(r_1_), static_cast<T>(r_2_), static_cast<T>(r_3_)};
    ceres::AngleAxisRotatePoint(r_vec, X1, X1_rotated);

    X1_RT[0] = X1_rotated[0] - t[0];
    X1_RT[1] = X1_rotated[1] - t[1];
    X1_RT[2] = X1_rotated[2] - t[2];

    residual[0] = (X2[0] - X1_RT[0])*(X2[0] - X1_RT[0]);
    residual[1] = (X2[1] - X1_RT[1])*(X2[1] - X1_RT[1]);
    residual[2] = (X2[2] - X1_RT[2])*(X2[2] - X1_RT[2]);

    return true;
}

void ba_spherical_costfunctor_tran_only::add_residual(Problem& problem
                         , vector<Point3d>& key_point_left_rect
                         , vector<Point3d>& key_point_right_rect
                         , double* init_rot
                         , double* init_tran
                         , vector<array<double, 2>>& init_d
                         , int match_num)
{
    for(int i = 0; i < match_num; i++)
    {
        CostFunction *cost_functor = new ceres::AutoDiffCostFunction<ba_spherical_costfunctor_tran_only, 3, 3>
                                    (new ba_spherical_costfunctor_tran_only(key_point_left_rect[i].x
                                                                , key_point_left_rect[i].y
                                                                , key_point_left_rect[i].z
                                                                , key_point_right_rect[i].x
                                                                , key_point_right_rect[i].y
                                                                , key_point_right_rect[i].z
                                                                , init_rot[0]
                                                                , init_rot[1]
                                                                , init_rot[2]
                                                                , init_d[0][0]
                                                                , init_d[1][0]));
        problem.AddResidualBlock(cost_functor, new ceres::HuberLoss(1.0), init_tran);
    }
}

// reprojection error
template <typename T> bool ba_spherical_costfunctor_d_only::operator()(const T *const d, T *residual) const
{
    // unit sphere projected point
    T X1[3], X2[3];
    X1[0] = cam1_x_*d[0];
    X1[1] = cam1_y_*d[0];
    X1[2] = cam1_z_*d[0];
    X2[0] = cam2_x_*d[1];
    X2[1] = cam2_y_*d[1];
    X2[2] = cam2_z_*d[1];

    // rotation and translation (R*Xn - t)
    T X1_rotated[3], X1_RT[3];
    T r_vec[3] = {static_cast<T>(r_1_), static_cast<T>(r_2_), static_cast<T>(r_3_)};
    ceres::AngleAxisRotatePoint(r_vec, X1, X1_rotated);

    X1_RT[0] = X1_rotated[0] - t_1_;
    X1_RT[1] = X1_rotated[1] - t_2_;
    X1_RT[2] = X1_rotated[2] - t_3_;

    residual[0] = (X2[0] - X1_RT[0])*(X2[0] - X1_RT[0]);
    residual[1] = (X2[1] - X1_RT[1])*(X2[1] - X1_RT[1]);
    residual[2] = (X2[2] - X1_RT[2])*(X2[2] - X1_RT[2]);
    residual[3] = lambda_*exp(-c_*d[0]);
    residual[4] = lambda_*exp(-c_*d[1]);

    return true;
}

void ba_spherical_costfunctor_d_only::add_residual(Problem& problem
                         , vector<Point3d>& key_point_left_rect
                         , vector<Point3d>& key_point_right_rect
                         , double* init_rot
                         , double* init_tran
                         , vector<array<double, 2>>& init_d
                         , int match_num)
{
    for(int i = 0; i < match_num; i++)
    {
        CostFunction *cost_functor = new ceres::AutoDiffCostFunction<ba_spherical_costfunctor_d_only, 5, 2>
                                    (new ba_spherical_costfunctor_d_only(key_point_left_rect[i].x
                                                                , key_point_left_rect[i].y
                                                                , key_point_left_rect[i].z
                                                                , key_point_right_rect[i].x
                                                                , key_point_right_rect[i].y
                                                                , key_point_right_rect[i].z
                                                                , init_tran[0]
                                                                , init_tran[1]
                                                                , init_tran[2]
                                                                , init_rot[0]
                                                                , init_rot[1]
                                                                , init_rot[2]
                                                                , 1.0
                                                                , 1.0));
        problem.AddResidualBlock(cost_functor, NULL, init_d[i].data());
        problem.SetParameterLowerBound(init_d[i].data(), 0, 0.0);
        problem.SetParameterLowerBound(init_d[i].data(), 1, 0.0);
    }
}