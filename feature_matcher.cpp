#include "feature_matcher.hpp"

using namespace std;
using namespace cv;

void feature_matcher::init()
{
    DEBUG_PRINT_OUT("initialize feature finder");
    START_TIME(initialize_feature_finder);

    ocl::setUseOpenCL(true);

    detector = xfeatures2d::SURF::create();
    //Ptr<xfeatures2d::BriefDescriptorExtractor> descriptor_extractor = xfeatures2d::BriefDescriptorExtractor::create(DESCRIPTER_SIZE, USE_ORIENTATION);
    descriptor_extractor = xfeatures2d::SURF::create();
    matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

    STOP_TIME(initialize_feature_finder);
}

void feature_matcher::deinit()
{

}

vector<KeyPoint> feature_matcher::detect_key_point(const Mat &image)
{
    vector<KeyPoint> key_point;
    detector->detect(image, key_point);

    return key_point;
}

Mat feature_matcher::comput_descriptor(const Mat &image, vector<KeyPoint> &key_point)
{
    Mat decriptors;
    descriptor_extractor->compute(image, key_point, decriptors);

    return decriptors;
}

vector<DMatch> feature_matcher::match_two_image(const Mat &descriptor1, const Mat &descriptor2)
{
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptor1, descriptor2, knn_matches, 2);

    const float ratio_thresh = 0.3f;
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

void feature_matcher::do_all(const Mat& im_left, const Mat& im_right, vector<KeyPoint>& left_key, vector<KeyPoint>& right_key, int& match_size, Mat& match_output)
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

    left_key = valid_key_left;
    right_key = valid_key_right;
    match_size = matches.size();
    match_output = outImage;
}
