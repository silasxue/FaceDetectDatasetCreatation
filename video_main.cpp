#include <iostream>
#include <vector>
#include <string>
#include <sstream>


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
using namespace std;


namespace fs = boost::filesystem;

void fromVectorToMat( vector<Rect> faceRects, Mat &f)
{
	f = Mat::zeros( faceRects.size(), 4, CV_32S);
	for( int c=0;c<faceRects.size();c++)
	{
		f.at<int>(c,0) = faceRects[c].x;
		f.at<int>(c,1) = faceRects[c].y;
		f.at<int>(c,2) = faceRects[c].width;
		f.at<int>(c,3) = faceRects[c].height;
	}
}


bool isSameTarget( Rect r1, Rect r2)
{

	/* 按照poscal 的标准是50%  交集/并集 > 0.5 */
	Rect intersect = r1 & r2;
	if(intersect.width * intersect.height < 1)
		return false;

	double union_area = r1.width*r1.height + r2.width*r2.height - intersect.width*intersect.height;

	/*  一个完全包含在另外一个里面，由于两种分类器训练原因 一大一小 */
	if( intersect == r1 && intersect.width*intersect.height/union_area >0.3 )
		return true;
	if( intersect == r2 && intersect.width*intersect.height/union_area >0.3 )
		return true;

	if( intersect.width*intersect.height/union_area < 0.5 )
		return false;

	return true;
}

void mergeAllTheFaces( vector<Rect> &pico_faces,
					   vector<Rect> &haar_faces,
					   vector<Rect> &hog_faces,
					   vector<Rect> &merge_faces)
{
	merge_faces.assign( pico_faces.begin(), pico_faces.end());

	/*  haar */
	for( int c=0;c<haar_faces.size();c++)
	{
		Rect newRect = haar_faces[c];
		bool already_have = false;
		for( int j=0;j<merge_faces.size();j++)
		{
			if( isSameTarget(newRect, merge_faces[j]) )
			{
				already_have = true;
				/* update the merge_faces, take the mean value */
				merge_faces[j].x = (merge_faces[j].x+newRect.x)/2;
				merge_faces[j].y = (merge_faces[j].y+newRect.y)/2;
				merge_faces[j].width = (merge_faces[j].width+newRect.width)/2;
				merge_faces[j].height = (merge_faces[j].height+newRect.height)/2;
				break;
			}
		}

		/* check */
		if(!already_have)
			merge_faces.push_back( newRect);
	}


	/*  hog */
	for( int c=0;c<hog_faces.size();c++)
	{
		Rect newRect = hog_faces[c];
		bool already_have = false;
		for( int j=0;j<merge_faces.size();j++)
		{
			if( isSameTarget(newRect, merge_faces[j]) )
			{
				already_have = true;
				merge_faces[j].x = (merge_faces[j].x+newRect.x)/2;
				merge_faces[j].y = (merge_faces[j].y+newRect.y)/2;
				merge_faces[j].width = (merge_faces[j].width+newRect.width)/2;
				merge_faces[j].height = (merge_faces[j].height+newRect.height)/2;
				break;
			}
		}

		/* check */
		if(!already_have)
			merge_faces.push_back( newRect);
	}

}

int main( int argc, char** argv)
{
	/* parameters */
	string video_path =  string(argv[1]);
	fs::path f_video_path(video_path);
	string video_name = fs::basename(video_path);

	string result_path = "../result/";
	string face_rec_path = "../face_recs/";
	int frame_skip = 5;
	int det_min_size = 30;
	int det_max_size = 400;
	double scale_factor = 1.05;

	VideoCapture cap(video_path);
	if(!cap.isOpened())
	{
		cout<<"can not open video, exit "<<endl;
		return -1;
	}

	/* 各种人脸检测方法初始化 */
	
	/* 1 haar */
	string face_cascade_name = "../xintai.xml";
	CascadeClassifier face_cascade;
	if(!face_cascade.load( face_cascade_name ) )
	{
		cout<<"can not load the face model "<<face_cascade_name<<endl;
		return -1;
	}

	/* 2 facedetWithFilter */
	FacedetWithFilter fff;
	fff.init("../svm_w.xml","../svm_b.xml");
	fff.setConfidenceParas( 3.0, 0.2, 0);			/* 设置比较低的阈值，以产生比较高的recall, 尽量不要错过 */
	fff.setDetectionParas( det_min_size, det_max_size, scale_factor,0.1);

	/*  3 fhog  */
	dlib::frontal_face_detector hog_detector = dlib::get_frontal_face_detector();
		
	Mat frame;
	int local_counter = 0;                      /* 抽桢 */
	int frame_counter = 0;                      /* 计数 */
	for(;;)
	{
		/* input image */
		cap >> frame;

		/*  header used by dlib */
		dlib::cv_image<dlib::bgr_pixel> input_im(frame);
		dlib::array2d<dlib::bgr_pixel> dlib_frame;
		dlib::assign_image( dlib_frame, input_im );

		/* 抽桢 */
		cout<<"local counter is "<<local_counter<<endl;
		if(!(local_counter++ == frame_skip))
			continue;
		local_counter = 0;
		
		
		frame_counter++;

		/* 检测人脸 */
		vector<Rect> faces_haar;
		vector<Rect> faces_ff;
		vector<Rect> faces_hog;

		fff.detectAndFilter( frame, faces_ff);
		face_cascade.detectMultiScale( frame, faces_haar, scale_factor, 2, 0, Size(det_min_size, det_min_size), Size(det_max_size, det_max_size));
		vector<dlib::rectangle> dets = hog_detector(dlib_frame);
		for (unsigned long j = 0; j < dets.size(); ++j)
        {
			int top_ = dets[j].top();
			int left_ = dets[j].left();
			int width_ = dets[j].width();
			int height_ = dets[j].height();
			Rect tmp = Rect( left_, top_, width_, height_ );
			faces_hog.push_back(tmp);
			//rectangle( cp3, tmp , Scalar(0,0,255));
		}


		/* 融合结果 */
		vector<Rect> merge_faces;
		mergeAllTheFaces( faces_ff, faces_haar, faces_hog, merge_faces );

		/* draw */
		Mat show; frame.copyTo(show);
		for( int c=0;c<faces_haar.size();c++)
			rectangle( show, faces_haar[c] , Scalar(0,0,255));
		for( int c=0;c<faces_ff.size();c++)
			rectangle( show, faces_ff[c] , Scalar(0,255,0));
		for( int c=0;c<faces_hog.size();c++)
			rectangle( show, faces_hog[c] , Scalar(255,0,0));
		for( int c=0;c<merge_faces.size();c++)
			rectangle( show, merge_faces[c] , Scalar(255,255,255));
		
		cout<<"result: haar \t"<<faces_haar.size()<<" detected "<<endl;
		cout<<"result: ff   \t"<<faces_ff.size()<<" detected "<<endl;
		cout<<"result: hog  \t"<<faces_hog.size()<<" detected "<<endl;
		cout<<"result: merge \t"<<merge_faces.size()<<" detected "<<endl;

		if(merge_faces.size() ==0)
			continue;

		/*  保存检测结果 1 图像 2 xml 3 人脸截图（用于纠错）*/

		/* 1  图像 */
		stringstream ss; ss<<frame_counter; string frame_string; ss>>frame_string;
		string image_file_name = video_name + "-" + frame_string+".jpg";
		cout<<"image file name is "<<image_file_name<<endl;
		imwrite( result_path+image_file_name, frame );

		/* 2 xml */
		FileStorage ffs( result_path+video_name + "-" + frame_string+".yml", FileStorage::WRITE);
		int number_of_detected_faces = merge_faces.size();
		ffs<<"number_of_face"<<number_of_detected_faces;
		Mat facerects;
		//fromVectorToMat( merge_faces, facerects);
		ffs<<"faces"<<merge_faces;
		ffs.release();


		for( int c=0; c<merge_faces.size();c++)
		{
			/* image name video_name-framecounter.jpg */
			stringstream ss; ss<<c; string face_string; ss>>face_string;
			string face_image_save_path = face_rec_path+video_name + "-" + frame_string+"-"+face_string+".jpg";
			imwrite( face_image_save_path, frame(merge_faces[c]));

		}

		imshow("inputFrame",show);
		waitKey(0);

	}
}
