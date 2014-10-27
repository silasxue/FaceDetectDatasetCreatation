#ifndef FACEDETWITHFILTER_H
#define FACEDETWITHFILTER_H

#include <iostream>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "FaceFilter.h"

using namespace cv;
using namespace std;

class FacedetWithFilter
{
	

	/************************************************************************/
	/*包括两部分--> facedetector 和 facefilter 的功能   
	/************************************************************************/
public:
	FacedetWithFilter();
	~FacedetWithFilter();

	/************************************************************************/
	/*  载入facefilter需要的 svm的w和b,facedetector不需要加载外部文件
		path_of_svm_w : in svm_w的路径 
		path_of_svm_b : in svm_b的路径                                                                     */
	/************************************************************************/
	
	bool init(  string path_of_svm_w,
			    string path_of_svm_b);

	
	/************************************************************************/
	/*  设置和阈值有关的参数，用于过滤人脸
		det_confidence	    : in  人脸检测的置信度
		color_filter_score : in  人脸过滤时候颜色阈值(人脸颜色所占的比例)
		hog_filter_score   : in  人脸过滤时候的形状(小于0不是，大于0是 数值绝对值越大，置信度越高)                                                                      */
	/************************************************************************/
	bool setConfidenceParas(	double det_confidence = 4.0,
								double color_filter_score = 0.3,
								double hog_filter_score = 0.1);


	/************************************************************************/
	/* 设置和人脸检测相关的参数
	   检测的参数会比较大的影响到速度：
	   ->最小人脸，最大人脸最好和现场的情况贴合，如果都是远景，就可以把最大人脸适当减少。
	   ->窗口的放缩参数要大于1，越小检测越细致，同时也越慢，但是一般1.2就满足要求
	   ->窗口滑动的比例也不要太小，应该是0-1之间的一个数字。
	   minsize			: in 最小检测人脸
	   maxsize			: in 最大检测人脸
	   scalefacor		: in 检测的放缩参数
	   stridefactor		: in 窗口滑动的距离占检测窗口的比例
	/************************************************************************/
	bool setDetectionParas( int minsize = 40,
							int maxsize = 90,
							double scalefactor = 1.2,
							double stridefactor = 0.2);

	/************************************************************************/
	/* 从图像中查找人脸，并且过滤人脸
		inputimage			: in  输入图像，最好是彩色，可以用color过滤人脸 
		passedfaces		    : out 检测到的人脸，都是通过了filter的                                                                      */
	/************************************************************************/
	bool detectAndFilter( const Mat &inputimage,   
						  vector<Rect> &passedfaces);

	/************************************************************************/
	/* 仅仅检测人脸
		inputimage			: in  输入图像，最好是彩色，可以用color过滤人脸 
		detectedfaces		: out 检测到的人脸                                                                     */
	/************************************************************************/
	bool detectFace(const Mat &inputimage, 
					vector<Rect> &detectedfaces);


	/************************************************************************/
	/* 仅仅过滤人脸
		infaces			: in  待过滤人脸  in-place                                                                      */
	/*  filtered_faces	: out 过滤后的人脸
	/*  image			: in  检测的图
	/************************************************************************/
	bool filterFace( vector<Rect> &infaces, 
					 vector<Rect> &filtered_faces,
					 const Mat &image);

	bool clean();
private:
	/*人脸检测部分的参数*/
	int		minsize;
	int		maxsize;
	double  scalefacor;
	double  stridefactor;
	double  fecedet_confidence;

	/*人脸过滤部分的参数*/
	double color_score;
	double hog_score;
	Mat svm_w;	// hog检测后的svm参数，由于是线性，可直接提取出
	Mat svm_b;

	/*色彩和颜色过滤器*/
	FaceFilter f_filter;

	/*人脸检测模型*/
	unsigned char* appfinder;
};
#endif