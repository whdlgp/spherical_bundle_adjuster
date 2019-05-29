#include "spherical_bundle_adjuster.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        DEBUG_PRINT_OUT("usage : spherical_bundle_adjuster.out <left image name> <right image name>");
        return 0;
    }
    
    String left_name = argv[1];
    String right_name = argv[2];

    Mat im_left = imread(left_name, IMREAD_COLOR);
    Mat im_right = imread(right_name, IMREAD_COLOR);

    spherical_bundle_adjuster sph_ba;
    sph_ba.do_all(im_left, im_right);
    return 0;
}
