#include "spherical_surf.hpp"
#define _USE_MATH_DEFINES
#include <cmath>

using namespace std;
using namespace cv;

void spherical_surf::set_omp(int num_proc)
{
    this->num_proc = num_proc;

    omp_set_num_threads(this->num_proc);
    DEBUG_PRINT_OUT("Number of process: " << this->num_proc);
}

// From OpenCV example utils.hpp code
// Calculates rotation matrix given euler angles.
Mat spherical_surf::eular2rot(Vec3f theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );
     
    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );
     
    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);
     
     
    // Combined rotation matrix
    Mat R = R_z * R_y * R_x;
     
    return R;
}

// rotate pixel, in_vec as input(row, col)
Vec2i spherical_surf::rotate_pixel(const Vec2i& in_vec, Vec3f theta, int width, int height)
{
    Mat rot_mat = eular2rot(theta);

    Vec2d vec_rad = Vec2d(M_PI*in_vec[0]/height, 2*M_PI*in_vec[1]/width);

    Vec3d vec_cartesian;
    vec_cartesian[0] = sin(vec_rad[0])*cos(vec_rad[1]);
    vec_cartesian[1] = sin(vec_rad[0])*sin(vec_rad[1]);
    vec_cartesian[2] = cos(vec_rad[0]);

    Vec3d vec_cartesian_rot;
    vec_cartesian_rot[0] = rot_mat.at<double>(0, 0)*vec_cartesian[0] + rot_mat.at<double>(0, 1)*vec_cartesian[1] + rot_mat.at<double>(0, 2)*vec_cartesian[2];
    vec_cartesian_rot[1] = rot_mat.at<double>(1, 0)*vec_cartesian[0] + rot_mat.at<double>(1, 1)*vec_cartesian[1] + rot_mat.at<double>(1, 2)*vec_cartesian[2];
    vec_cartesian_rot[2] = rot_mat.at<double>(2, 0)*vec_cartesian[0] + rot_mat.at<double>(2, 1)*vec_cartesian[1] + rot_mat.at<double>(2, 2)*vec_cartesian[2];

    Vec2d vec_rot;
    vec_rot[0] = acos(vec_cartesian_rot[2]);
    vec_rot[1] = atan2(vec_cartesian_rot[1], vec_cartesian_rot[0]);
    if(vec_rot[1] < 0)
        vec_rot[1] += M_PI*2;

    Vec2i vec_pixel;
    vec_pixel[0] = height*vec_rot[0]/M_PI;
    vec_pixel[1] = width*vec_rot[1]/(2*M_PI);

    return vec_pixel;
}

Mat spherical_surf::crop_rotated_image(float pitch_rot, const Mat& im)
{
    int im_height = im.rows;
    int im_width = im.cols;

    Mat out(im_height/4, im_width, im.type());

    Mat2i im_pixel_rotate(im_height/4, im_width);
    
    #pragma omp parallel for
    for(int i = 0; i < im_height/4; i++)
    {
        for(int j = 0; j < im_width; j++)
        {
            int offset_i = i+im_height*3/8;
            // inverse warping
            Vec2i vec_pixel = rotate_pixel(Vec2i(offset_i, j) 
                                         , Vec3f(0
                                               , RAD(pitch_rot)
                                               , 0)
                                         , im_width, im_height);

            int out_i = vec_pixel[0];
            int out_j = vec_pixel[1];
            if((out_i >= 0) && (out_j >= 0) && (out_i < im_height) && (out_j < im_width))
            {
                out.at<Vec3b>(i, j) = im.at<Vec3b>(out_i, out_j);
            }
        }
    }

    return out;
}

void spherical_surf::rotate_keypoint(float pitch_rot_inv, vector<KeyPoint>& key, int width, int height)
{
    #pragma omp parallel for
    for(int i = 0; i < key.size(); i++)
    {
        int offset_i = key[i].pt.y+height*3/8;
        Vec2i vec_pixel = rotate_pixel(Vec2i(offset_i, key[i].pt.x) 
                                     , Vec3f(0
                                           , RAD(pitch_rot_inv)
                                           , 0)
                                     , width, height);
        key[i].pt.x = vec_pixel[1];
        key[i].pt.y = vec_pixel[0];
    }
}

void spherical_surf::do_all(const Mat& im_left, const Mat& im_right, vector<KeyPoint>& left_key, vector<KeyPoint>& right_key, int& match_size, Mat& match_output)
{
    int im_width = im_left.cols;
    int im_height = im_left.rows;

    int offset_x = 0;
    int offset_y = im_height*3/8;
    Rect roi(offset_x, offset_y, im_width, im_height/4);

    // Rotate and Crop, to reduce occlusion because of projection
    DEBUG_PRINT_OUT("left image,");
    DEBUG_PRINT_OUT("Rotate ROLL to 45, Crop Undistorted resion");
    Mat left_n0 = crop_rotated_image(45, im_left);
    DEBUG_PRINT_OUT("Rotate ROLL to 0, Crop Undistorted resion");
    Mat left_n1 = im_left(roi);
    DEBUG_PRINT_OUT("Rotate ROLL to -45, Crop Undistorted resion");
    Mat left_n2 = crop_rotated_image(-45, im_left);
    DEBUG_PRINT_OUT("Rotate ROLL to -90, Crop Undistorted resion");
    Mat left_n3 = crop_rotated_image(-90, im_left);

    DEBUG_PRINT_OUT("right image,");
    DEBUG_PRINT_OUT("Rotate ROLL to 45, Crop Undistorted resion");
    Mat right_n0 = crop_rotated_image(45, im_right);
    DEBUG_PRINT_OUT("Rotate ROLL to 0, Crop Undistorted resion");
    Mat right_n1 = im_right(roi);
    DEBUG_PRINT_OUT("Rotate ROLL to -45, Crop Undistorted resion");
    Mat right_n2 = crop_rotated_image(-45, im_right);
    DEBUG_PRINT_OUT("Rotate ROLL to -90, Crop Undistorted resion");
    Mat right_n3 = crop_rotated_image(-90, im_right);

    // Find features and make descriptor to each n
    feature_matcher fm;
    DEBUG_PRINT_OUT("Find features of left image");
    vector<KeyPoint> key_left_n0 = fm.detect_key_point(left_n0);
    vector<KeyPoint> key_left_n1 = fm.detect_key_point(left_n1);
    vector<KeyPoint> key_left_n2 = fm.detect_key_point(left_n2);
    vector<KeyPoint> key_left_n3 = fm.detect_key_point(left_n3);

    DEBUG_PRINT_OUT("Find features of right image");
    vector<KeyPoint> key_right_n0 = fm.detect_key_point(right_n0);
    vector<KeyPoint> key_right_n1 = fm.detect_key_point(right_n1);
    vector<KeyPoint> key_right_n2 = fm.detect_key_point(right_n2);
    vector<KeyPoint> key_right_n3 = fm.detect_key_point(right_n3);

    DEBUG_PRINT_OUT("Comput descriptor");
    Mat desc_left_n0 = fm.comput_descriptor(left_n0, key_left_n0);
    Mat desc_left_n1 = fm.comput_descriptor(left_n1, key_left_n1);
    Mat desc_left_n2 = fm.comput_descriptor(left_n2, key_left_n2);
    Mat desc_left_n3 = fm.comput_descriptor(left_n3, key_left_n3);

    Mat desc_right_n0 = fm.comput_descriptor(right_n0, key_right_n0);
    Mat desc_right_n1 = fm.comput_descriptor(right_n1, key_right_n1);
    Mat desc_right_n2 = fm.comput_descriptor(right_n2, key_right_n2);
    Mat desc_right_n3 = fm.comput_descriptor(right_n3, key_right_n3);
    DEBUG_PRINT_OUT("descriptor size " << desc_left_n0.rows << ", " << desc_left_n0.cols);

    DEBUG_PRINT_OUT("Rotate found key point");
    rotate_keypoint(45, key_left_n0, im_width, im_height);
    #pragma omp parallel for
    for(int i = 0; i < key_left_n1.size(); i++)
        key_left_n1[i].pt.y = key_left_n1[i].pt.y + im_height*3/8;
    rotate_keypoint(-45, key_left_n2, im_width, im_height);
    rotate_keypoint(-90, key_left_n3, im_width, im_height);

    rotate_keypoint(45, key_right_n0, im_width, im_height);
    #pragma omp parallel for
    for(int i = 0; i < key_right_n1.size(); i++)
        key_right_n1[i].pt.y = key_right_n1[i].pt.y + im_height*3/8;
    rotate_keypoint(-45, key_right_n2, im_width, im_height);
    rotate_keypoint(-90, key_right_n3, im_width, im_height);

    DEBUG_PRINT_OUT("Concatenate key points");
    left_key_tmp.insert(left_key_tmp.end(), key_left_n0.begin(), key_left_n0.end());
    left_key_tmp.insert(left_key_tmp.end(), key_left_n1.begin(), key_left_n1.end());
    left_key_tmp.insert(left_key_tmp.end(), key_left_n2.begin(), key_left_n2.end());
    left_key_tmp.insert(left_key_tmp.end(), key_left_n3.begin(), key_left_n3.end());

    right_key_tmp.insert(right_key_tmp.end(), key_right_n0.begin(), key_right_n0.end());
    right_key_tmp.insert(right_key_tmp.end(), key_right_n1.begin(), key_right_n1.end());
    right_key_tmp.insert(right_key_tmp.end(), key_right_n2.begin(), key_right_n2.end());
    right_key_tmp.insert(right_key_tmp.end(), key_right_n3.begin(), key_right_n3.end());

    DEBUG_PRINT_OUT("Concatenate descriptors");
    Mat desc_left_arr[] = {desc_left_n0, desc_left_n1, desc_left_n2, desc_left_n3};
    vconcat(desc_left_arr, 4, descriptor_left_tmp);
    Mat desc_right_arr[] = {desc_right_n0, desc_right_n1, desc_right_n2, desc_right_n3};
    vconcat(desc_right_arr, 4, descriptor_right_tmp);

    DEBUG_PRINT_OUT("Match descriptor and leave only valid key points");
    matches = fm.match_two_image(descriptor_left_tmp, descriptor_right_tmp);

    vector<KeyPoint> valid_key_left(matches.size());
    vector<KeyPoint> valid_key_right(matches.size());
    #pragma omp parallel for
    for(int i = 0; i < matches.size(); i++)
    {
        valid_key_left[i] = left_key_tmp[matches[i].queryIdx];
        valid_key_right[i] = right_key_tmp[matches[i].trainIdx];
    }    
    
    // For test imshow
    vector<DMatch> tmp_match(matches.size());
    #pragma omp parallel for
    for(int i = 0; i < matches.size(); i++)
    {
        tmp_match[i].queryIdx = i;
        tmp_match[i].trainIdx = i;
        tmp_match[i].distance = matches[i].distance;
    }
    Mat outImage = fm.draw_match(im_left, im_right, valid_key_left, valid_key_right);
    
    left_key = valid_key_left;
    right_key = valid_key_right;
    match_size = matches.size();
    match_output = outImage;
}