#include <iostream>
#include <vector>
#include <string>


#include "boost/filesystem.hpp"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

namespace fs = boost::filesystem;


void fromMatToVector( const Mat &inputMat, vector<Rect> &faceRects )
{
	for ( int c=0;c<inputMat.rows ;c++ )
	{
		faceRects.push_back( Rect( inputMat.at<int>(c,1), inputMat.at<int>(c,2),inputMat.at<int>(c,3),inputMat.at<int>(c,4)));
	}
	
}

void parse_name( const string &full_name,
				 string &file_name, 
				 string &faceid)
{
	int last_dot = full_name.find_last_of(".");
	int last_dash = full_name.find_last_of("-");

	faceid = full_name.substr( last_dash+1, last_dot - last_dash );
	file_name = full_name.substr(0, last_dash);
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  main
 *  Description:  
 * =====================================================================================
 */
int
main ( int argc, char *argv[] )
{
	if(argc !=2)
	{
		cout<<"format should be : ./show_result path_to_the_result_folder/ "<<endl;
		return -1;
	}

	bool removeTheNoFaceImage = true;

	string data_path_   = argv[1];				    	/* 原始的结果存放的文件夹 */

	fs::path data_path( data_path_ );

	if(!fs::exists(data_path))
	{
		std::cout<<"data_path "<<data_path<<" not exist "<<std::endl;
		return -1;
	}

	if(!fs::is_directory(data_path))
	{
		std::cout<<"data_path is not a directory "<<data_path<<std::endl;
		return -1;
	}

	/* processing .. */
	fs::directory_iterator end_it;
	for( fs::directory_iterator file_iter(data_path); file_iter != end_it; file_iter++ )
    {
		/* 根据名称定位yml文件和需要删除的face的id(从0开始的) */
		fs::path s = *(file_iter);
        string pathname = file_iter->path().string();
		string basename = fs::basename(s);
		string extname  = fs::extension(s);

		if( extname != ".yml")
		{
			continue;
		}

	
		/* read the yml file  */
		string yml_file_ = data_path_+basename+".yml";
		string img_file_ = data_path_+basename+".jpg";
		Mat input_image = imread( img_file_);

		fs::path yml_file(yml_file_);
		fs::path img_file(img_file_);

		if(!fs::exists(yml_file))
		{
			cout<<"error, file "<<yml_file_<<" does not exist !"<<endl;
			return -1;
		}

		FileStorage ffs( yml_file_ , FileStorage::READ);
		if(!ffs.isOpened())
		{
			cout<<"error, can not open "<<yml_file_<<" for FileStorage open"<<endl;
			return -1;

		}
		
		cout<<"processing file "<<img_file_<<endl;

		int number_of_face;
		vector<Rect> facerects;
		Mat faceMat;
		ffs["number_of_face"]>>number_of_face;
		ffs["faces"]>>faceMat;
		fromMatToVector(faceMat, facerects);

		/* 删除那些不包含人脸的图像 */
		if(removeTheNoFaceImage && facerects.size()==0)
		{
			cout<<"delete file "<<img_file_<<endl;
			fs::remove( yml_file);
			fs::remove( img_file);
		}


		ffs.release();

		for(int c=0;c<facerects.size();c++)
		{
			rectangle( input_image, facerects[c], Scalar(0,255,255), 3);
		}

		/* show */
		resize( input_image ,input_image, Size(0,0), 0.6, 0.6);
		putText( input_image, "image name:"+basename, Point(40,40),FONT_HERSHEY_COMPLEX, 0.8, Scalar(0,0,0), 2 );
		imshow( "result", input_image);
		waitKey(0);

	}
	return 0;
}				/* ----------  end of function main  ---------- */
