#include "../spherical_surf.hpp"
#include "../equi2cube_surf.hpp"
#include "../debug_print.h"

#include <sstream>
#include <fstream>
#include <numeric>

#include <omp.h>

using namespace std;
using namespace cv;

Vec3d rad2cart(const Vec2d& vec_rad)
{
    Vec3d vec_cartesian;
    vec_cartesian[0] = sin(vec_rad[0])*cos(vec_rad[1]);
    vec_cartesian[1] = sin(vec_rad[0])*sin(vec_rad[1]);
    vec_cartesian[2] = cos(vec_rad[0]);
    return vec_cartesian;
}

double get_dist(Vec3d& v1, Vec3d& v2)
{
    Vec3d diff = v1 - v2;
    return sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
}

double get_angle(Vec3d& v1, Vec3d& v2)
{
    float in_product = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    double angle = acos(in_product);
    return angle;
}

double get_diff(const KeyPoint& key_left, const KeyPoint& key_right, Mat& rot_mat, int width, int height)
{
    Vec2d left_rad = Vec2d(M_PI*key_left.pt.y/height, 2*M_PI*key_left.pt.x/width);
    Vec2d right_rad = Vec2d(M_PI*key_right.pt.y/height, 2*M_PI*key_right.pt.x/width);
    Vec3d left_cart = rad2cart(left_rad);
    Vec3d right_cart = rad2cart(right_rad);

    double* rot_mat_data = (double*)rot_mat.data;
    Vec3d left_cart_rot;
    left_cart_rot[0] = rot_mat_data[0]*left_cart[0] + rot_mat_data[1]*left_cart[1] + rot_mat_data[2]*left_cart[2];
    left_cart_rot[1] = rot_mat_data[3]*left_cart[0] + rot_mat_data[4]*left_cart[1] + rot_mat_data[5]*left_cart[2];
    left_cart_rot[2] = rot_mat_data[6]*left_cart[0] + rot_mat_data[7]*left_cart[1] + rot_mat_data[8]*left_cart[2];
    double diff = abs(get_angle(left_cart_rot, right_cart));
    diff = fmod(diff, 2*M_PI);

    /*
    spherical_surf ss;
    Vec2i left_key_rot = ss.rotate_pixel(Vec2i(key_left.pt.y, key_left.pt.x)
                                           , rot_mat
                                           , width, height);

    double diff = sqrt((key_right.pt.y - left_key_rot[0]) * (key_right.pt.y - left_key_rot[0])
                         + (key_right.pt.x - left_key_rot[1]) * (key_right.pt.x - left_key_rot[1]));
    */

    return diff;
}

void draw_test_match(const Mat& im_left, const Mat& im_right, const vector<KeyPoint>& key_left, const vector<KeyPoint>& key_right,
                    double expected_roll, double expected_pitch, double expected_yaw, double threshold, int total_key_num, String logname)
{
    Mat im_right_copy = im_right.clone();
    
    spherical_surf ss;
    vector<double> diffs(key_left.size());
    vector<Point2f> left_key_rot;
    int outlier = 0;
    Mat rot_mat = ss.eular2rot(Vec3f(RAD(expected_roll), RAD(expected_pitch), RAD(expected_yaw)));
    for(int i = 0; i < key_left.size(); i++)
    {
        diffs[i] = get_diff(key_left[i], key_right[i], rot_mat, im_left.cols, im_left.rows);

        Vec2i tmp = ss.rotate_pixel(Vec2i(key_left[i].pt.y, key_left[i].pt.x)
                                           , rot_mat
                                           , im_left.cols, im_left.rows);
        left_key_rot.push_back(Point2f(tmp[1], tmp[0]));

        Scalar val;
        if(diffs[i] <= threshold)
        {
            val = Scalar(0, 255, 0);
            line(im_right_copy, left_key_rot[i], key_right[i].pt, val, 3);
        }
    }

    for(int i = 0; i < key_left.size(); i++)
    {
        Scalar val;
        if(diffs[i] > threshold)
        {
            outlier++;
            val = Scalar(0, 0, 255);
            line(im_right_copy, left_key_rot[i], key_right[i].pt, val, 3);
        }
    }

    sort(diffs.begin(), diffs.end());
    int ten_percent_size = diffs.size()*0.1;
    vector<double> diffs_mid(diffs.begin()+ten_percent_size, diffs.end()-ten_percent_size);
    double diff_mean = std::accumulate(diffs_mid.begin(), diffs_mid.end(), 0.0)/(diffs_mid.size()*1.0);

    String test_dir = "test/";
    String test_result_dir =  test_dir+"test_result/";
    
    ofstream log_test_file;
    log_test_file.open(test_result_dir+logname+".txt", std::ios_base::app);
    log_test_file << expected_roll << ","
                  << expected_pitch << ","
                  << expected_yaw << ","
                  << key_left.size() << ","
                  << outlier << ","
                  << (outlier*100.0)/(key_left.size()*1.0) << ","
                  << total_key_num << ","
                  << diff_mean
                  << endl;
    log_test_file.close();

    string file_name;
    stringstream file_name_stream(file_name);
    file_name_stream << test_result_dir << logname;
    file_name_stream << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ".JPG" << endl;
    file_name_stream >> file_name;
    DEBUG_PRINT_OUT("name: " << file_name);
    imwrite(file_name, im_right_copy);
}

int main(int argc, char** argv)
{
    
    if(argc != 3)
    {
        DEBUG_PRINT_OUT("usage : E_matrix_test.out <left image name> <right image name>");
        return 0;
    }
    
    // Parse input
    String left_name = argv[1];
    String right_name = argv[2];
    double expected_roll, expected_pitch, expected_yaw;

    vector<string> input_parse;
    string s = argv[2];
    string delimiter = "_";

    size_t pos = 0;
    string token;
    while((pos = s.find(delimiter)) != std::string::npos) 
    {
        input_parse.push_back(s.substr(0, pos));
        s.erase(0, pos + delimiter.length());
    }
    pos = s.find(".");
    input_parse.push_back(s.substr(0, pos));
    s.erase(0, pos + delimiter.length());

    expected_roll = stod(input_parse[1]);
    expected_pitch = stod(input_parse[2]);
    expected_yaw = stod(input_parse[3]);

    Mat im_left = imread(left_name, IMREAD_COLOR);
    Mat im_right = imread(right_name, IMREAD_COLOR);

    vector<KeyPoint> left_key_fm;
    vector<KeyPoint> right_key_fm;
    int match_size_fm;
    Mat match_output_fm;
    int total_key_num_fm;

    DEBUG_PRINT_OUT("Do feature finding and matching in ERP, normal surf");
    feature_matcher fm;
    fm.do_all(im_left, im_right, left_key_fm, right_key_fm, match_size_fm, match_output_fm, total_key_num_fm);

    vector<KeyPoint> left_key_ss;
    vector<KeyPoint> right_key_ss;
    int match_size_ss;
    Mat match_output_ss;
    int total_key_num_ss;
    
    DEBUG_PRINT_OUT("Do feature finding and matching in ERP, proposed surf");
    spherical_surf ss;
    ss.set_omp(omp_get_num_procs());
    ss.do_all(im_left, im_right, left_key_ss, right_key_ss, match_size_ss, match_output_ss, total_key_num_ss);

    vector<KeyPoint> left_key_es;
    vector<KeyPoint> right_key_es;
    int match_size_es;
    Mat match_output_es;
    int total_key_num_es;

    DEBUG_PRINT_OUT("Do feature finding and matching with cubemap");
    equi2cube_surf es;
    es.set_omp(omp_get_num_procs());
    es.set_cube_size(600);
    es.do_all(im_left, im_right, left_key_es, right_key_es, match_size_es, match_output_es, total_key_num_es);

    String logname_fm = left_name;
    logname_fm += "_fm";
    String logname_ss = left_name;
    logname_ss += "_ss"; 
    String logname_es = left_name;
    logname_es += "_es"; 

    double th = 2.0/180.0*M_PI; // 2 degree as threshold
    draw_test_match(im_left, im_right, left_key_fm, right_key_fm, expected_roll, expected_pitch, expected_yaw, th, total_key_num_fm, logname_fm);
    draw_test_match(im_left, im_right, left_key_ss, right_key_ss, expected_roll, expected_pitch, expected_yaw, th, total_key_num_ss, logname_ss);
    draw_test_match(im_left, im_right, left_key_es, right_key_es, expected_roll, expected_pitch, expected_yaw, th, total_key_num_es, logname_es);

    return 0;
}
