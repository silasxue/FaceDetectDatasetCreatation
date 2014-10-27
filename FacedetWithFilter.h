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
	/*����������--> facedetector �� facefilter �Ĺ���   
	/************************************************************************/
public:
	FacedetWithFilter();
	~FacedetWithFilter();

	/************************************************************************/
	/*  ����facefilter��Ҫ�� svm��w��b,facedetector����Ҫ�����ⲿ�ļ�
		path_of_svm_w : in svm_w��·�� 
		path_of_svm_b : in svm_b��·��                                                                     */
	/************************************************************************/
	
	bool init(  string path_of_svm_w,
			    string path_of_svm_b);

	
	/************************************************************************/
	/*  ���ú���ֵ�йصĲ��������ڹ�������
		det_confidence	    : in  �����������Ŷ�
		color_filter_score : in  ��������ʱ����ɫ��ֵ(������ɫ��ռ�ı���)
		hog_filter_score   : in  ��������ʱ�����״(С��0���ǣ�����0�� ��ֵ����ֵԽ�����Ŷ�Խ��)                                                                      */
	/************************************************************************/
	bool setConfidenceParas(	double det_confidence = 4.0,
								double color_filter_score = 0.3,
								double hog_filter_score = 0.1);


	/************************************************************************/
	/* ���ú����������صĲ���
	   ���Ĳ�����Ƚϴ��Ӱ�쵽�ٶȣ�
	   ->��С���������������ú��ֳ���������ϣ��������Զ�����Ϳ��԰���������ʵ����١�
	   ->���ڵķ�������Ҫ����1��ԽС���Խϸ�£�ͬʱҲԽ��������һ��1.2������Ҫ��
	   ->���ڻ����ı���Ҳ��Ҫ̫С��Ӧ����0-1֮���һ�����֡�
	   minsize			: in ��С�������
	   maxsize			: in ���������
	   scalefacor		: in ���ķ�������
	   stridefactor		: in ���ڻ����ľ���ռ��ⴰ�ڵı���
	/************************************************************************/
	bool setDetectionParas( int minsize = 40,
							int maxsize = 90,
							double scalefactor = 1.2,
							double stridefactor = 0.2);

	/************************************************************************/
	/* ��ͼ���в������������ҹ�������
		inputimage			: in  ����ͼ������ǲ�ɫ��������color�������� 
		passedfaces		    : out ��⵽������������ͨ����filter��                                                                      */
	/************************************************************************/
	bool detectAndFilter( const Mat &inputimage,   
						  vector<Rect> &passedfaces);

	/************************************************************************/
	/* �����������
		inputimage			: in  ����ͼ������ǲ�ɫ��������color�������� 
		detectedfaces		: out ��⵽������                                                                     */
	/************************************************************************/
	bool detectFace(const Mat &inputimage, 
					vector<Rect> &detectedfaces);


	/************************************************************************/
	/* ������������
		infaces			: in  ����������  in-place                                                                      */
	/*  filtered_faces	: out ���˺������
	/*  image			: in  ����ͼ
	/************************************************************************/
	bool filterFace( vector<Rect> &infaces, 
					 vector<Rect> &filtered_faces,
					 const Mat &image);

	bool clean();
private:
	/*������ⲿ�ֵĲ���*/
	int		minsize;
	int		maxsize;
	double  scalefacor;
	double  stridefactor;
	double  fecedet_confidence;

	/*�������˲��ֵĲ���*/
	double color_score;
	double hog_score;
	Mat svm_w;	// hog�����svm���������������ԣ���ֱ����ȡ��
	Mat svm_b;

	/*ɫ�ʺ���ɫ������*/
	FaceFilter f_filter;

	/*�������ģ��*/
	unsigned char* appfinder;
};
#endif