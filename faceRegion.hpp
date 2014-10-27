#ifndef FACEREGION_HPP
#define FACEREGION_HPP

#include <stdio.h>
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"

#include <iostream>
using namespace std;

struct TCbCr
{
	double Cb;
	double Cr;
} CbCr;

struct TCbCr CalCbCr(int B, int G, int R)
{
	struct TCbCr res;
	res.Cb =( 128 - 37.797 * R/255 - 74.203 * G/255 +   112 * B/255);
	res.Cr =( 128 + 112    * R/255 - 93.786 * G/255 -18.214 * B/255);
	return res;
}

	double bmean=117.4361;//��˹ģ�Ͳ���
	double rmean=156.5599;
	double brcov[2][2]={160.1301,12.1430,12.1430,299.4574} ;

/*********************�˲�*****************************/

void filter(double **source,int m_nWidth,int m_nHeight)
{
	int x,y;
	double **temp;
	//����һ����ʱ��ά��???
	temp = new  double*[m_nHeight+2];
	for(x=0;x <=m_nHeight+1; x++)
		temp[x] = new double[m_nWidth+2];

	//�߽����???
	for(x=0; x<=m_nHeight+1; x++)
	{
		temp[x][0] = 0;
		temp[x][m_nWidth+1] = 0;
	}
	for(y=0; y<=m_nWidth+1; y++)
	{
		temp[0][y] = 0;
		temp[m_nHeight+1][y] = 0;
	}

	//��ԭ�����ֵ������ʱ��???
	for(x=0; x<m_nHeight; x++)
		for(y=0; y<m_nWidth; y++)
			temp[x+1][y+1] = source[x][y];

	//��ֵ��???
	for(x=0; x<m_nHeight; x++)
	{
		for(y=0; y<m_nWidth; y++)
		{
			source[x][y] = 0;
			for(int k=0;k<=2;k++)
				for(int l=0;l<=2;l++)
					source[x][y] += temp[x+k][y+l];

			source[x][y] /= 9;
		}
	}

	if(temp!=NULL)
	{
		for(int x=0;x<=m_nHeight+1;x++)
            if(temp[x]!=NULL)
                delete []temp[x];
        delete []temp;
	}
}
/**********************otsu�㷨��???**************************/
int otsuThreshold(IplImage *frame)
{

    int width = frame->width;
	int height = frame->height;
	int pixelCount[256];
	float pixelPro[256];
	int i, j, pixelSum = width * height, threshold = 0;
	uchar* data = (uchar*)frame->imageData;

	for(i = 0; i <256; i++)
	{
		pixelCount[i] = 0;
		pixelPro[i] = 0;
	}

	//ͳ�ƻҶȼ���ÿ������������ͼ���еĸ�???
	for(i = 0; i < height; i++)
	{
		for(j = 0;j < width;j++)
		{
		pixelCount[(int)data[i * frame->widthStep+ j]]++;
		}
	}

	//����ÿ������������ͼ���еı�???
	for(i = 0; i < 256; i++)
	{
		pixelPro[i] = (float)pixelCount[i] / pixelSum;
	}

	//�����Ҷȼ�[0,255]
	float w0, w1, u0tmp, u1tmp, u0, u1, u, 
			deltaTmp, deltaMax = 0;
	for(i = 0; i < 256; i++)
	{
		w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
		for(j = 0; j < 256; j++)
		{
			if(j <= i)   //��������
			{
				w0 += pixelPro[j];
				u0tmp += j * pixelPro[j];
			}
			else   //ǰ������
			{
				w1 += pixelPro[j];
				u1tmp += j * pixelPro[j];
			}
		}
		u0 = u0tmp / w0;
		u1 = u1tmp / w1;
		u = u0tmp + u1tmp;
		deltaTmp = 
			w0 * pow((u0 - u), 2) + w1 * pow((u1 - u), 2);
		if(deltaTmp > deltaMax)
		{
			deltaMax = deltaTmp;
			threshold = i;
		}
	}
	return threshold;
}

/*******************************�򵥸�˹ģ???********************************************/
IplImage* likeliHood(IplImage *pImg)
{
    double **m_pLikeliHoodArray = new double*[pImg->height+1];
    for(int c=0;c<pImg->height+1;c++)
        m_pLikeliHoodArray[c] = new double[pImg->width+1];

	int pImgH=pImg->height;
	int pImgW=pImg->width;

    for(int i=0; i<pImgH; i++)     //����Ycbcr�ռ�ļ򵥸�˹��???
	{
		for(int j=0; j<pImgW; j++)
		{
			double x1,x2;    


            int t_b = ((uchar *)(pImg->imageData + i*pImg->widthStep))[j*pImg->nChannels + 0];
            int t_g = ((uchar *)(pImg->imageData + i*pImg->widthStep))[j*pImg->nChannels + 1];
            int t_r = ((uchar *)(pImg->imageData + i*pImg->widthStep))[j*pImg->nChannels + 2];
            TCbCr temp = CalCbCr(t_b,t_g,t_r);

			x1 = temp.Cb-bmean;
			x2 = temp.Cr-rmean;
			  double t;
			t = x1*(x1*brcov[1][1]-x2*brcov[1][0])+x2*(-x1*brcov[0][1]+x2*brcov[0][0]);
			t /= (brcov[0][0]*brcov[1][1]-brcov[0][1]*brcov[1][0]);
			t /= (-2);
			m_pLikeliHoodArray[i][j] = exp(t);
		}
	}

    filter(m_pLikeliHoodArray,pImgW,pImgH);

	double max = 0.0;
	for(int i=0; i<pImgH; i++)
		for(int j=0; j<pImgW; j++)
			if(m_pLikeliHoodArray[i][j] > max) 
				max = m_pLikeliHoodArray[i][j];
	
	for(int i=0; i<pImgH; i++)
	{
		for(int j=0; j<pImgW; j++)
		{
			m_pLikeliHoodArray[i][j] /= max;
            m_pLikeliHoodArray[i][j]=m_pLikeliHoodArray[i][j]*255;
		}
	}

    IplImage *imgGauss = 0;
	imgGauss = cvCreateImage(cvGetSize(pImg),IPL_DEPTH_8U,1);
    CvScalar imgTemp;
	 
	for(int a=0;a<pImg->height;a++)
	{
		 for(int b=0;b<pImg->width;b++)
		 {
			imgTemp = cvGet2D(imgGauss,a,b);
			imgTemp.val[0] = m_pLikeliHoodArray[a][b];
            cvSet2D(imgGauss,a,b,imgTemp);
		}

	}

    for(int c=0;c<pImg->height+1;c++)
    {
        delete []m_pLikeliHoodArray[c];
    }
	delete[]m_pLikeliHoodArray;

	return imgGauss;

}

/****************************************OTSU******************************************************/
double OTSU_ratio(IplImage* imgGauss)
{
     int threValue;
     threValue = otsuThreshold(imgGauss);//����otsu������???
   
	 CvScalar otsu;
	 int m;
	 
	int countIsFace = 0;

	 for(int i=0;i<imgGauss->height;i++)//  ��ֵ��
	 {
		 for(int j=0;j<imgGauss->width;j++)
		 {
			 otsu=cvGet2D(imgGauss,i,j);
			 m=(int)otsu.val[0];
			 if(m>=threValue)
			 {
				 countIsFace++;
			 }
			 else
			 {

			 }
		 }
	 }
	 double ratio = countIsFace/(1.0*imgGauss->height*imgGauss->width);
	 return ratio;
}


    
// API
double getFaceRegionRatio( IplImage *inputImage )
{
	IplImage* imgGauss(0);
   
	imgGauss=likeliHood(inputImage);
    double ratio = OTSU_ratio(imgGauss);

	if(imgGauss)
		cvReleaseImage(&imgGauss);
	return ratio;
}

#endif
 
