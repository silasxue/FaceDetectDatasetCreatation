#include <fstream>
#include <stdio.h>
#include <vector>
#include "FacedetWithFilter.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "picort.h"



bool FacedetWithFilter::clean()
{
	//
	return true;
}


FacedetWithFilter::FacedetWithFilter()
{
	vector<unsigned char> tmp_hold;         // 纯文本文件，第一次读取的时候也无法知道具体的长度
	
    FILE *ff = fopen("../facefinder_pure.ea","r");
	unsigned int  tmp_c;
	while ( !feof(ff))
	{
		fscanf(ff,"%x",&tmp_c);
		tmp_hold.push_back(tmp_c);
	}
	fclose(ff);
	
	this->appfinder = new unsigned char[tmp_hold.size()];
	for (int c=0;c<tmp_hold.size();c++)
	{
		this->appfinder[c] = tmp_hold[c];
	}

	// 使用默认参数，函数自带默认参数
	this->setConfidenceParas(4.0,0.3,0.1);
	this->setDetectionParas(40,90,1.2,0.2);
}

FacedetWithFilter::~FacedetWithFilter()
{
	if(!this->appfinder)
		delete this->appfinder;
}

bool FacedetWithFilter::setDetectionParas(int _minsize ,
										  int _maxsize ,
										  double _scalefactor ,
										  double _stridefactor)
{
	if (_minsize < 0 ||_maxsize < 0 ||(_minsize > _maxsize))
	{
		return false;
	}

	if( _scalefactor < 1 || _stridefactor > 1)
	{
		return false;
	}

	this->minsize = _minsize;
	this->maxsize = _maxsize;
	this->scalefacor = _scalefactor;
	this->stridefactor = _stridefactor;
	
	return true;
}

bool FacedetWithFilter::setConfidenceParas(	double det_confidence ,
											double color_filter_score ,
											double hog_filter_score)
{
	this->fecedet_confidence = det_confidence;
	this->color_score = color_filter_score;
	this->hog_score = hog_filter_score;

	return true;
}

bool FacedetWithFilter::init( string path_of_svm_w,
							  string path_of_svm_b)
{
	//过滤器初始化
	if(!f_filter.init(path_of_svm_w,path_of_svm_b))
		return false;
	return true;
}


bool FacedetWithFilter::filterFace( vector<Rect> &infaces,
								    vector<Rect> &filtered_faces,
								    const Mat &image)
{
	if( infaces.size() == 0)
	{
		cout<<"empty input faces"<<endl;
		return false;
	}

	for( int c=0;c<infaces.size();c++)
	{
		if ( f_filter.filter( image(infaces[c]), this->color_score, this->hog_score))
		{
			filtered_faces.push_back( infaces[c]);
		}
	}

	return true;
}


bool FacedetWithFilter::detectFace( const Mat &inputimage, 
									vector<Rect> &detectedfaces)
{
	const IplImage fframe = inputimage.operator _IplImage();
	const IplImage *frame = &fframe;

	int i;
	unsigned char* pixels;
	int nrows, ncols, ldim;

	const int MAXNDETECTIONS =2048;
	int ndetections;
	float qs[MAXNDETECTIONS], rs[MAXNDETECTIONS], cs[MAXNDETECTIONS], ss[MAXNDETECTIONS];

	IplImage* gray = 0;

	

	// grayscale image
	if(!gray)
		gray = cvCreateImage(cvSize(frame->width, frame->height), frame->depth, 1);
	if(frame->nChannels == 3)
		cvCvtColor(frame, gray, CV_RGB2GRAY);
	else
		cvCopy(frame, gray, 0);

	// get relevant image data
	pixels = (unsigned char*)gray->imageData;
	nrows = gray->height;
	ncols = gray->width;
	ldim = gray->widthStep;

	// actually, all the smart stuff happens here
	// 0 -15 15 30 -30
	ndetections = find_objects(0.0f, rs, cs, ss, qs, MAXNDETECTIONS, this->appfinder, pixels, nrows, ncols, ldim, this->scalefacor, this->stridefactor, minsize, maxsize, 0);
    
	//ndetections += find_objects( 2*3.14f/360*(360-10) , &rs[ndetections], &cs[ndetections], &ss[ndetections], &qs[ndetections], MAXNDETECTIONS, this->appfinder, pixels, nrows, ncols, ldim, this->scalefacor,this->stridefactor,minsize, maxsize, 0);
	//ndetections += find_objects( 2*3.14f/360*(10) , &rs[ndetections], &cs[ndetections], &ss[ndetections], &qs[ndetections], MAXNDETECTIONS, this->appfinder, pixels, nrows, ncols, ldim, this->scalefacor,this->stridefactor,minsize, maxsize, 0);
	//ndetections += find_objects( 2*3.14f/360*(360-30) , &rs[ndetections], &cs[ndetections], &ss[ndetections], &qs[ndetections], MAXNDETECTIONS, this->appfinder, pixels, nrows, ncols, ldim, this->scalefacor,this->stridefactor,minsize, maxsize, 0);
	//ndetections += find_objects( 2*3.14f/360*(30) , &rs[ndetections], &cs[ndetections], &ss[ndetections], &qs[ndetections], MAXNDETECTIONS, this->appfinder, pixels, nrows, ncols, ldim, this->scalefacor,this->stridefactor,minsize, maxsize, 0);

	//ndetections = cluster_detections(rs, cs, ss, qs, ndetections);


	std::vector<double> det_weight;

	for(int i=0;i<ndetections;i++)
	{
		if(qs[i] > this-> fecedet_confidence)
		{
			Rect facerec( cs[i] - ss[i]/2, rs[i] -ss[i]/2, ss[i], ss[i] );
			detectedfaces.push_back( facerec);
			det_weight.push_back( qs[i] );
		}
	}


	// 2014, 10 ,15 group the detections
	cv::groupRectangles( detectedfaces, 1, 0.3);


	if(gray)
		cvReleaseImage(&gray);

	return true;
}

bool FacedetWithFilter::detectAndFilter(const Mat &inputimage,   
									    vector<Rect> &passedfaces)
{
	vector<Rect> faces;
	if(!this->detectFace(inputimage, faces))
	{
		return false;
	}

	if( faces.empty())
	{
		return true;
	}

	if(!this->filterFace( faces, passedfaces, inputimage))
	{
		return false;
	}

	return true;
}
