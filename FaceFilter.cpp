#include <string>
#include <iostream>
#include <vector>

#include "FaceFilter.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "faceRegion.hpp"

using namespace std;
using namespace cv;

FaceFilter::FaceFilter():hog(Size(40,40), Size(16,16),Size(8,8), Size(8,8), 9)
{

}

bool FaceFilter::init( string path_of_svm_w, 
					   string path_of_svm_b)
{
    FileStorage fs;
	fs.open( path_of_svm_w, FileStorage::READ);
    if(!fs.isOpened())
    {
        cout<<"can not load the model "<<path_of_svm_w<<endl;
        return false;
    }
    fs["matrix"]>>this->svm_w;
    fs.release();

    fs.open(path_of_svm_b, FileStorage::READ);
    if(!fs.isOpened())
    {
        cout<<"can not load the model "<<path_of_svm_b<<endl;
        return false;
    }
    fs["matrix"]>>this->svm_b;
    fs.release();

    return true;
}

bool FaceFilter::filter( const Mat &inputimage, 
						 double faceRegionThrehold, 
						 double hogFilterThrehold)
{
    /* color filter first, then the HOG filter
     */

    /* 1 check the input image, shoule be a RGB image, else just skip the color filter step
     */

  

    if( inputimage.channels()!= 1)
    {
        IplImage img = IplImage(inputimage);
        double colorScore = getFaceRegionRatio(&img);
        // can not pass the color region test
        cout<<"the color score is "<<colorScore<<endl;
        if(colorScore < faceRegionThrehold)
		{
			cout<<"filter because color "<<endl;
            return false;
		}
    }
    else
    {
        cout<<"skipped the color filter step"<<endl;
    }




    /* 2 check the hog filter score
     */

    //get the hog feature

    Mat temp_img;
    resize(inputimage, temp_img, Size(40,40));
    vector<float> des;
	des.reserve(600);
    this->hog.compute( temp_img, des);


    double final_score = 0;
    for(int c=0;c<des.size();c++)
    { 
		final_score += svm_w.at<double>(c,0)*des[c];
    }
    final_score += svm_b.at<double>(0,0);

  
    cout<<"hog score is "<<final_score<<endl;
    if(final_score < hogFilterThrehold)
	{
		cout<<"filter because hog , less than "<<hogFilterThrehold<<endl;
        return false;
	}

   
    // pass all the test
    return true;
}
