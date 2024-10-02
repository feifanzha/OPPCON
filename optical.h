# pragma once

# include <opencv2/opencv.hpp>
# include <eigen3/Eigen/Core>
# include<eigen3/Eigen/Dense>

extern std::vector<float> f_x;
extern std::vector<float> f_y;
struct PlaneModel{//ax+by+cz+d = 0
    double a;
    double b;
    double c;
    double d;
};
std::vector<cv::Point2f> Get8Pix(const cv::Point2f &keypoints_1,std::vector<cv::Point2f> &pix_surround);
std::vector<cv::Point2f> diff(const std::vector<cv::Point2f>& vec1,const std::vector<cv::Point2f> &vec2,const float camera_fx,const float camera_fy);
std::vector<cv::Point2f> OFCal(const std::vector<cv::Point2f> &pixels,const cv::Mat &frame1, const cv::Mat &frame2,std::vector<uchar> &status,float camera_fx,float camera_fy);
float WeightSum(const std::vector<cv::Point2f> &neighbours ,const int pos, const std::vector<float> &weight,int locate);
float GraudOp(const std::vector<cv::Point2f> &neighbors, const int pos);
bool WhetherChase(const std::vector<cv::Point2f> &neighbors,const std::vector<uchar> & status, const int pos);
std::vector<cv::Point3f> Pixel23D(std::vector<cv::Point2f> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2,float camera_fx,float camera_fy);
double point2PlaneDistance(const cv::Point3f& point, const PlaneModel& plane);
std::vector<cv::Point2f> estimatePlaneRansac(const std::vector<cv::Point3f> &points,int  max_iterations,double threshold);
void estimatePlaneRansac(const std::vector<cv::Point3f> &points, std::vector<cv::Point2f> &static_kps,std::vector<cv::Point2f> &dynamic_kps,int  max_iterations,double threshold);
void KeypointDynamicOpticalDef(std::vector<cv::Point2f> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2,std::vector<cv::Point2f> &static_kps, std::vector<cv::Point2f> &dynamic_kps, double threshold,float camera_fx = 1.0F,float camera_fy = 1.0F);
std::vector<cv::Point2f> OpticalDefStatic(std::vector<cv::Point2f> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2, double threshold,float camera_fx = 1.0F,float camera_fy = 1.0F);
void DisplayDynamic(std::vector<cv::Point2f> &Dynamic_kps,std::vector<cv::Point2f> &Static_kps,cv::Mat &img1);
void savePicture(std::vector<cv::Point2f> &Dynamic_kps,std::vector<cv::Point2f> &Static_kps,cv::Mat &img1,const std::string &path);
std::vector<float> Pixel23D(std::vector<cv::KeyPoint> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2,float camera_fx = 1.0F,float camera_fy = 1.0F);
double point2PlaneDistance(const cv::Point2f& pointxy, float kpz, PlaneModel& plane);
void estimatePlaneRansac(const std::vector<cv::KeyPoint> &pointxy,const std::vector<float> &kpz, std::vector<cv::KeyPoint> &static_kps,std::vector<cv::KeyPoint> &dynamic_kps,int  max_iterations,double threshold);
std::vector<cv::KeyPoint> estimatePlaneRansac(const std::vector<cv::KeyPoint> &pointxy,std::vector<float> &kps_z,int  max_iterations,double threshold);
void KeypointDynamicOpticalDef(std::vector<cv::KeyPoint> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2,std::vector<cv::KeyPoint> &static_kps, std::vector<cv::KeyPoint> &dynamic_kps, double threshold,float camera_fx = 1.0F,float camera_fy = 1.0F);
std::vector<cv::KeyPoint> OpticalDefStatic(std::vector<cv::KeyPoint> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2, double threshold,float camera_fx = 1.0F,float camera_fy = 1.0F);
void DisplayDynamic(std::vector<cv::KeyPoint> &Dynamic_kps,std::vector<cv::KeyPoint> &Static_kps,cv::Mat &img1);
void savePicture(std::vector<cv::KeyPoint> &Dynamic_kps,std::vector<cv::KeyPoint> &Static_kps,cv::Mat &img1,const std::string &path);
