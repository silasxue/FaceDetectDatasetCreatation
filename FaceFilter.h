#ifndef FACEFILTER_H
#define FACEFILTER_H

#include <string>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ocl/ocl.hpp"

using namespace cv;
using namespace std;


class FaceFilter
{
public:
    FaceFilter();
    bool init( string path_of_svm_w, string path_of_svm_b );
    bool filter( const Mat &inputimage, double faceRegionThrehold, double hogFilterThrehold);
    void clean();

private:
    Mat svm_w;
    Mat svm_b;
    HOGDescriptor hog;
};


#endif // FACEFILTER_H
