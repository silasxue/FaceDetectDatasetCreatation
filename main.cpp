#include <iostream>
#include <vector>
#include <string>


#include "boost/filesystem.hpp"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>


#include "FacedetWithFilter.h"

using namespace cv;


namespace fs = boost::filesystem;

int main( int argc, char** argv)
{

	string face_cascade_name = "../xintai.xml";
	CascadeClassifier face_cascade;

	if(!face_cascade.load( face_cascade_name ) )
	{
		std::cout<<"can not load the face model "<<std::endl;
		return -1;
	}

	string image_path = "/home/yuanyang/workspace/face_align_one_millisecond/face_landmark_data/lfpw/trainset/";
	fs::path input_path( image_path );

	if(!fs::exists(input_path))
	{
		std::cout<<"input_path "<<input_path<<" not exist "<<std::endl;
	}

	if(!fs::is_directory(input_path))
	{
		std::cout<<"input_path is not a directory "<<input_path<<std::endl;
	}

	
	FacedetWithFilter fff;
	fff.init("../svm_w.xml","../svm_b.xml");
	
    fff.setConfidenceParas( 2.0, 0.1, -10);		//²ÎÊýÒâÒåŒûÀà¶šÒå


	/* load face detector from dlib */
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	
	
	fs::directory_iterator end_it;
	for( fs::directory_iterator file_iter(input_path); file_iter != end_it; file_iter++ )
    {
		fs::path s = *(file_iter);
		
        string basename = fs::basename( s );
        string pathname = file_iter->path().string();
		string extname  = fs::extension( *(file_iter) );
		if( extname!=".jpg" &&
			extname!=".png")
		{
            //std::cout<<"do no support ext name "<<extname<<std::endl;
			continue;
		}

		Mat im = imread( pathname );

		dlib::cv_image<dlib::bgr_pixel> input_im(im);
		
		dlib::array2d<dlib::bgr_pixel> img;
		dlib::assign_image( img, input_im );
		
		int min_ = 50;
		int max_ = min( im.cols, im.rows);
		double scale_factor = 1.05;

        fff.setDetectionParas( min_ , max_ , scale_factor, 0.1);	//Œì²âµÄ²ÎÊý
		
		vector<Rect> faces;
		faces.reserve(10);

		Mat cp1,cp2,cp3;
		im.copyTo(cp1);
		im.copyTo(cp2);
		im.copyTo(cp3);


		fff.setConfidenceParas(2.5, 0.2, -10);

		double t = cv::getTickCount();
        fff.detectAndFilter( im , faces);
	//	fff.detectFace(im , faces );
		t = (double)cv::getTickCount() - t;
		std::cout<<"time duration pico det and filter is "<<t/cv::getTickFrequency()<<std::endl;

		t = cv::getTickCount();
		fff.detectFace(im , faces );
		t = (double)cv::getTickCount() - t;
		std::cout<<"time duration pico det is "<<t/cv::getTickFrequency()<<std::endl;


		for(int c=0;c<faces.size();c++)
		{
			rectangle( cp1, faces[c], Scalar(255,0,0));
		}
		imshow("pico", cp1 );
		
		/* haar boost 
		 * */
	
		vector<Rect> faces2;
	
		t = (double)getTickCount();
		face_cascade.detectMultiScale( im, faces2, scale_factor, 2, 0, Size(min_,min_), Size(max_,max_) );
		t = (double)getTickCount() - t;
		std::cout<<"time duration haarboost is "<<t/getTickFrequency()<<std::endl;

		for(int c=0;c<faces2.size();c++)
		{
			rectangle( cp2, faces2[c], Scalar(0,255,0));
		}
		imshow("haar", cp2 );



		/* dlib face detect */
		t = (double)cv::getTickCount();
        std::vector<dlib::rectangle> dets = detector(img);
		t = (double)cv::getTickCount() - t;
		std::cout<<"time duration in dlib is "<<t/(double)cv::getTickFrequency()<<std::endl;

		for (unsigned long j = 0; j < dets.size(); ++j)
        {
			int top_ = dets[j].top();
			int left_ = dets[j].left();
			int width_ = dets[j].width();
			int height_ = dets[j].height();

			rectangle( cp3, Rect( left_, top_, width_, height_ ), Scalar(0,0,255));
		}

		imshow("shog",cp3);


		waitKey(0);	
	}
      
	return 0;
}
