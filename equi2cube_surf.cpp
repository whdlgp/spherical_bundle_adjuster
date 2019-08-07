#include "equi2cube_surf.hpp"

using namespace std;
using namespace cv;

void equi2cube_surf::set_omp(int num_proc)
{
    this->num_proc = num_proc;

    omp_set_num_threads(this->num_proc);
    DEBUG_PRINT_OUT("Number of process: " << this->num_proc);
}

void equi2cube_surf::set_cube_size(int cube_size)
{
    this->cube_size = cube_size;
}

void equi2cube_surf::cube2equi_pixel(Point2f& cube_pixel, Point2f& equi_pixel, int cube_size, int im_width, int im_height)
{
    Vec3d vec_cart;
    if(cube_pixel.x < cube_size) //left
    {
        vec_cart[0] = (cube_size - 2.0*cube_pixel.x)/cube_size;
        vec_cart[1] = 1.0;
        vec_cart[2] = (cube_size - 2.0*cube_pixel.y)/cube_size;
    }
    else if(cube_pixel.x >= cube_size && cube_pixel.x < 2*cube_size) //front
    {
        vec_cart[0] = -1.0;
        vec_cart[1] = (cube_size - 2.0*(cube_pixel.x - cube_size))/cube_size;
        vec_cart[2] = (cube_size - 2.0*cube_pixel.y)/cube_size;
    }
    else if(cube_pixel.x >= 2*cube_size && cube_pixel.x < 3*cube_size) //right
    {
        vec_cart[0] = (2.0*(cube_pixel.x - 2*cube_size) - cube_size)/cube_size;
        vec_cart[1] = -1.0;
        vec_cart[2] = (cube_size - 2.0*cube_pixel.y)/cube_size;
    }
    else if(cube_pixel.x >= 3*cube_size && cube_pixel.x < 4*cube_size) //back
    {
        vec_cart[0] = 1.0;
        vec_cart[1] = (2.0*(cube_pixel.x - 3*cube_size) - cube_size)/cube_size;
        vec_cart[2] = (cube_size - 2.0*cube_pixel.y)/cube_size;
    }
    else if(cube_pixel.x >= 4*cube_size && cube_pixel.x < 5*cube_size) //top
    {
        vec_cart[0] = (cube_size - 2.0*cube_pixel.y)/cube_size;
        vec_cart[1] = (cube_size - 2.0*(cube_pixel.x - 4*cube_size))/cube_size;
        vec_cart[2] = 1.0;
    }
    else if(cube_pixel.x >= 5*cube_size) //bottom
    {
        vec_cart[0] = (2.0*cube_pixel.y - cube_size)/cube_size;
        vec_cart[1] = (cube_size - 2.0*(cube_pixel.x - 5*cube_size))/cube_size;
        vec_cart[2] = -1.0;
    }

    // to unit vector
    Vec3d vec_unit_cart;
    double vec_norm = sqrt(vec_cart[0]*vec_cart[0] + vec_cart[1]*vec_cart[1] + vec_cart[2]*vec_cart[2]);
    vec_unit_cart[0] = vec_cart[0]/vec_norm;
    vec_unit_cart[1] = vec_cart[1]/vec_norm;
    vec_unit_cart[2] = vec_cart[2]/vec_norm;

    // to radian
    Vec2d vec_rad;
    vec_rad[0] = acos(vec_unit_cart[2]);
    vec_rad[1] = atan2(vec_unit_cart[1], vec_unit_cart[0]);
    if(vec_rad[1] < 0)
        vec_rad[1] += M_PI*2;

    // to pixel
    equi_pixel.x = im_width*vec_rad[1]/(2*M_PI);
    equi_pixel.y = im_height*vec_rad[0]/M_PI;
}

void equi2cube_surf::do_all(const Mat& im_left, const Mat& im_right, vector<KeyPoint>& left_key, vector<KeyPoint>& right_key, int& match_size, Mat& match_output, int& total_key_num)
{
    int im_width = im_left.cols;
    int im_height = im_left.rows;

    equi2cube cube;
    cube.set_omp(omp_get_num_procs());
    Mat cubemap1 = cube.get_all(im_left, cube_size);
    Mat cubemap2 = cube.get_all(im_right, cube_size);

    feature_matcher fm;
    DEBUG_PRINT_OUT("Find and Match features");
    vector<KeyPoint> key_left_cube = fm.detect_key_point(cubemap1);
    vector<KeyPoint> key_right_cube = fm.detect_key_point(cubemap2);
    Mat desc_left = fm.comput_descriptor(cubemap1, key_left_cube);
    Mat desc_right = fm.comput_descriptor(cubemap2, key_right_cube);
    vector<DMatch> matches = fm.match_two_image(desc_left, desc_right);

    vector<KeyPoint> key_left_equi(key_left_cube.size());
    vector<KeyPoint> key_right_equi(key_left_cube.size());
    for(int i = 0; i < key_left_equi.size(); i++)
    {
        key_left_equi[i] = key_left_cube[i];
        key_right_equi[i] = key_right_cube[i];
        cube2equi_pixel(key_left_cube[i].pt, key_left_equi[i].pt, cube_size, im_width, im_height);
        cube2equi_pixel(key_right_cube[i].pt, key_right_equi[i].pt, cube_size, im_width, im_height);
    }

    DEBUG_PRINT_OUT("Draw features");
    vector<KeyPoint> valid_key_left(matches.size());
    vector<KeyPoint> valid_key_right(matches.size());
    for(int i = 0; i < matches.size(); i++)
    {
        valid_key_left[i] = key_left_equi[matches[i].queryIdx];
        valid_key_right[i] = key_right_equi[matches[i].trainIdx];
    }

    Mat outImage = fm.draw_match(im_left, im_right, valid_key_left, valid_key_right);

    left_key = valid_key_left;
    right_key = valid_key_right;
    match_size = matches.size();
    match_output = outImage;
    total_key_num = key_left_cube.size();
}
