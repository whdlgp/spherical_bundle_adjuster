#include "spherical_bundle_adjuster.hpp"

using namespace std;
using namespace cv;

void spherical_bundle_adjuster::init()
{
    DEBUG_PRINT_OUT("initialize stitcher");
    START_TIME(initialize_stitcher);

    ocl::setUseOpenCL(true);

    detector = xfeatures2d::SURF::create();
    //Ptr<xfeatures2d::BriefDescriptorExtractor> descriptor_extractor = xfeatures2d::BriefDescriptorExtractor::create(DESCRIPTER_SIZE, USE_ORIENTATION);
    descriptor_extractor = xfeatures2d::SURF::create();
    matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    STOP_TIME(initialize_stitcher);
}

void spherical_bundle_adjuster::deinit()
{

}

vector<KeyPoint> spherical_bundle_adjuster::detect_key_point(const Mat &image)
{
    vector<KeyPoint> key_point;
    detector->detect(image, key_point);

    return key_point;
}

Mat spherical_bundle_adjuster::comput_descriptor(const Mat &image, vector<KeyPoint> &key_point)
{
    Mat decriptors;
    descriptor_extractor->compute(image, key_point, decriptors);

    return decriptors;
}

vector<DMatch> spherical_bundle_adjuster::match_two_image(const Mat &descriptor1, const Mat &descriptor2)
{
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);

    const float ratio_thresh = 0.4f;
    std::vector<DMatch> good_matches;

    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    return good_matches;
}

using ceres::CENTRAL;
using ceres::CostFunction;
using ceres::NumericDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

// spherical bundle adjustment
struct ba_spherical_costfunctor
{
    ba_spherical_costfunctor(double cam1_x, double cam1_y, double cam1_z, double cam2_x, double cam2_y, double cam2_z)
    : cam1_x_(cam1_x), cam1_y_(cam1_y), cam1_z_(cam1_z), cam2_x_(cam2_x), cam2_y_(cam2_y), cam2_z_(cam2_z)
    {}

    // reprojection error
    template <typename T> bool operator()(const T *const d, const T *const r, const T *const t, T *residual) const
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

    private:
    const double cam1_x_;
    const double cam1_y_;
    const double cam1_z_;
    const double cam2_x_;
    const double cam2_y_;
    const double cam2_z_;
};

void spherical_bundle_adjuster::do_all(const Mat& im_left, const Mat& im_right)
{
    //Finding features, making descriptor
    START_TIME(Feature_finding_make_descriptor);
    key_point_left = detect_key_point(im_left);
    key_point_right = detect_key_point(im_right);
    descriptor_left = comput_descriptor(im_left, key_point_left);
    descriptor_right = comput_descriptor(im_right, key_point_right);
    STOP_TIME(Feature_finding_make_descriptor);

    //Matching features
    START_TIME(Matching);
    matches = match_two_image(descriptor_left, descriptor_right);
    DEBUG_PRINT_OUT("matched : " << matches.size());
    STOP_TIME(Matching);

    // leave only valide KeyPoints
    vector<KeyPoint> valid_key_left(matches.size());
    vector<KeyPoint> valid_key_right(matches.size());
    for(int i = 0; i < matches.size(); i++)
    {
        valid_key_left[i] = key_point_left[matches[i].queryIdx];
        valid_key_right[i] = key_point_right[matches[i].trainIdx];
    }

    // For test imshow
    vector<DMatch> tmp_match(matches.size());
    for(int i = 0; i < matches.size(); i++)
    {
        tmp_match[i].queryIdx = i;
        tmp_match[i].trainIdx = i;
        tmp_match[i].distance = matches[i].distance;
    }
    Mat outImage;
    cv::drawMatches(im_left, valid_key_left, im_right, valid_key_right, tmp_match, outImage);
    cv::imwrite("test_match.png", outImage);

    // convert pixel to radian coordinate, in unit sphere
    // x : longitude
    // y : latitude
    double im_width = im_left.cols;
    double im_height = im_left.rows;
    vector<Point2d> key_point_left_radian(matches.size());
    vector<Point2d> key_point_right_radian(matches.size());
    for(int i = 0; i < matches.size(); i++)
    {
        key_point_left_radian[i].x = 2*M_PI*(valid_key_left[i].pt.x / im_width);
        key_point_right_radian[i].x = 2*M_PI*(valid_key_right[i].pt.x / im_width);
        key_point_left_radian[i].y = M_PI*(valid_key_left[i].pt.y / im_height);
        key_point_right_radian[i].y = M_PI*(valid_key_right[i].pt.y / im_height);
    }

    // convert radian to rectangular coordinate
    vector<Point3d> key_point_left_rect(matches.size());
    vector<Point3d> key_point_right_rect(matches.size());
    for(int i = 0; i < matches.size(); i++)
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

    vector<array<double, 2>> init_d(matches.size());
    for(int i = 0; i < matches.size(); i++)
    {
        init_d[i][0] = 1000;
        init_d[i][1] = 1000;
    }
    double init_rot[3] = {0.0, 0.0, 0.0};
    double init_tran[3] = {1000.0, 0.0, 0.0};

    for(int i = 0; i < matches.size(); i++)
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

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.max_num_iterations = 50;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";

    std::cout << "rotation vector " << init_rot[0] << ' ' << init_rot[1] << ' ' << init_rot[2]<< std::endl;
    std::cout << "translation vector " << init_tran[0] << ' ' << init_tran[1] << ' ' << init_tran[2] << std::endl;

    Mat d_image;
    d_image = im_left.clone();
    circle(d_image, valid_key_left[0].pt, 10, Scalar(255, 0, 0), 5);
    cv::imwrite("d_image.png", d_image);

    for(int i = 0; i < matches.size(); i++)
    {

        std::cout << init_d[i][0] << ',' << init_d[i][1] << std::endl;
    }
    
    DEBUG_PRINT_OUT("Done."); 
}