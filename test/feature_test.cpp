#include "../spherical_surf.hpp"
#include "../equi2cube_surf.hpp"
#include "../debug_print.h"

#include <sstream>
#include <fstream>

#include <omp.h>

using namespace std;
using namespace cv;

void draw_test_match(const Mat& im_left, const Mat& im_right, const vector<KeyPoint>& key_left, const vector<KeyPoint>& key_right,
                    double expected_roll, double expected_pitch, double expected_yaw, double threshold, int total_key_num, String logname)
{
    Mat im_left_gray, im_right_gray;
    cvtColor(im_left, im_left_gray, CV_RGB2GRAY);
    cvtColor(im_right, im_right_gray, CV_RGB2GRAY);

    Mat im_overlap(im_left.rows, im_left.cols, im_left.type());
    Mat chan[3];
    chan[0] = im_left_gray;
    chan[1] = im_right_gray;
    chan[2] = Mat::zeros(im_left.rows, im_left.cols, CV_8UC1);
    merge(chan, 3, im_overlap);
    
    spherical_surf ss;
    vector<double> diffs(key_left.size());
    int outlier = 0;
    Mat rot_mat = ss.eular2rot(Vec3f(RAD(expected_roll), RAD(expected_pitch), RAD(expected_yaw)));
    for(int i = 0; i < key_left.size(); i++)
    {
        Vec2i left_key_rot = ss.rotate_pixel(Vec2i(key_left[i].pt.y, key_left[i].pt.x)
                                           , rot_mat
                                           , im_left.cols, im_left.rows);
        diffs[i] = sqrt((key_right[i].pt.y - left_key_rot[0]) * (key_right[i].pt.y - left_key_rot[0])
                         + (key_right[i].pt.x - left_key_rot[1]) * (key_right[i].pt.x - left_key_rot[1]));

        Scalar val;
        if(diffs[i] <= threshold)
        {
            val = Scalar(0, 255, 0);
            line(im_overlap, key_left[i].pt, key_right[i].pt, val, 3);
        }
    }

    for(int i = 0; i < key_left.size(); i++)
    {
        Scalar val;
        if(diffs[i] > threshold)
        {
            outlier++;
            val = Scalar(0, 0, 255);
            line(im_overlap, key_left[i].pt, key_right[i].pt, val, 3);
        }
    }

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
                  << total_key_num
                  << endl;
    log_test_file.close();

    string file_name;
    stringstream file_name_stream(file_name);
    file_name_stream << test_result_dir << logname;
    file_name_stream << expected_roll << ',' << expected_pitch << ',' << expected_yaw << ".JPG" << endl;
    file_name_stream >> file_name;
    DEBUG_PRINT_OUT("name: " << file_name);
    imwrite(file_name, im_overlap);
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
    draw_test_match(im_left, im_right, left_key_fm, right_key_fm, expected_roll, expected_pitch, expected_yaw, 22.5, total_key_num_fm, logname_fm);
    draw_test_match(im_left, im_right, left_key_ss, right_key_ss, expected_roll, expected_pitch, expected_yaw, 22.5, total_key_num_ss, logname_ss);
    draw_test_match(im_left, im_right, left_key_es, right_key_es, expected_roll, expected_pitch, expected_yaw, 22.5, total_key_num_es, logname_es);

    return 0;
}
