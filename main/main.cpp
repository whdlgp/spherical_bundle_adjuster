#include "../spherical_bundle_adjuster.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if(argc != 10)
    {
        DEBUG_PRINT_OUT("usage : spherical_bundle_adjuster.out <L image> <R image> <exp roll> <exp pitch> <exp yaw> <exp Tx> <exp Ty> <exp Tz> <exp d>");
        return 0;
    }
    
    String left_name = argv[1];
    String right_name = argv[2];

    Mat im_left = imread(left_name, IMREAD_COLOR);
    Mat im_right = imread(right_name, IMREAD_COLOR);

    double expected_roll, expected_pitch, expected_yaw, expected_tx, expected_ty, expected_tz, expected_d;
    expected_roll = atof(argv[3]);
    expected_pitch = atof(argv[4]);
    expected_yaw = atof(argv[5]);
    expected_tx = atof(argv[6]);
    expected_ty = atof(argv[7]);
    expected_tz = atof(argv[8]);
    expected_d = atof(argv[9]);

    spherical_bundle_adjuster sph_ba(expected_roll, expected_pitch, expected_yaw
                                     , expected_tx, expected_ty, expected_tz, expected_d);
    sph_ba.set_omp(omp_get_num_procs());
    sph_ba.do_bundle_adjustment(im_left, im_right);
    return 0;
}
