#include "spherical_bundle_adjuster.hpp"

using namespace std;
using namespace cv;

int main()
{
    Mat im_left = imread("left.JPG", IMREAD_COLOR);
    Mat im_right = imread("right.JPG", IMREAD_COLOR);

    spherical_bundle_adjuster sph_ba;
    sph_ba.do_all(im_left, im_right);
    return 0;
}
