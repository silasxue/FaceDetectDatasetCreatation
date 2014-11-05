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

void deleteById( Mat &ori, int id,  Mat &output)
{
	for ( int c=0;c<ori.rows ;c++ ) 
	{
		if(id == ori.at<int>(c,0))
			continue;
		output.push_back( ori.row(c));
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
	if(argc != 3)
	{
		cout<<"format should be :./revise_result path_to_wrong_face_folder/ path_to_result_folder/"<<endl;
		return -1;
	}

	string result_path_ = argv[1];                   /* 错误的人脸图片存放的文件夹 */
	string data_path_   = argv[2];				    	/* 原始的结果存放的文件夹 */

	fs::path result_path( result_path_ );
	fs::path data_path( data_path_ );
	
	/*  check  */
	if(!fs::exists(result_path))
	{
		std::cout<<"result_path "<<result_path<<" not exist "<<std::endl;
		return -1;
	}

	if(!fs::is_directory(result_path))
	{
		std::cout<<"result_path is not a directory "<<result_path<<std::endl;
		return -1;
	}

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
	for( fs::directory_iterator file_iter(result_path); file_iter != end_it; file_iter++ )
    {
		/* 根据名称定位yml文件和需要删除的face的id(从0开始的) */
		fs::path s = *(file_iter);
        string pathname = file_iter->path().string();
		string basename = fs::basename(s);
		string extname  = fs::extension(s);
		cout<<"processing "<<basename<<" with extension "<<extname<<endl;
		if( extname != ".jpg")
		{
			cout<<"skip file "<<pathname<<endl;
			continue;
		}

		string faceid, filename;
		parse_name(basename, filename, faceid);

		/* read the yml file  */
		string yml_file_ = data_path_+filename+".yml";
		fs::path yml_file(yml_file_);
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
		
		int number_of_face;
		Mat faceMat;
		ffs["number_of_face"]>>number_of_face;
		ffs["faces"]>>faceMat;
		ffs.release();

		/*  删除 faceid 对应的那一个 */
		stringstream ss;ss<<faceid;int id_face;ss>>id_face;
		Mat newFaceMat;
		deleteById( faceMat, id_face, newFaceMat);
		number_of_face--;

		/* 删除原来的yml文件, 写入新的yml*/
		fs::remove(yml_file);
		fs::remove(pathname);							/* 注意这里也删除拉原来的错误图片 避免一个图片用两次 造成错误 */
		ffs.open( yml_file_, FileStorage::WRITE);
		ffs<<"number_of_face"<<number_of_face;
		ffs<<"faces"<<newFaceMat;
		ffs.release();

	}




	return 0;
}				/* ----------  end of function main  ---------- */
