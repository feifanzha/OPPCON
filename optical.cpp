# include "optical.h"

std::vector<float> f_x = {-1.0,0.0,1.0,-2.0,2.0,-1.0,0.0,1.0};
std::vector<float> f_y = {1.0,2.0,1.0,0.0,0.0,-1.0,-2.0,-1.0};

std::vector<cv::Point2f> Get8Pix(const cv::Point2f &keypoints_1,std::vector<cv::Point2f> &pix_surround)
{
    for(int i = -1; i <= 1 ;i++){
        for(int j = -1; j<= 1; j++){
            if(i == 0 && j == 0) continue;
            cv::Point2f neighbor(keypoints_1.x+i,keypoints_1.y+j);
            pix_surround.push_back(neighbor);
        }
    }
    return pix_surround;
}

std::vector<cv::Point2f> diff(const std::vector<cv::Point2f>& vec1,const std::vector<cv::Point2f> &vec2,float camera_fx = 1, float camera_fy = 1)
{
    std::vector<cv::Point2f> result;
    result.reserve(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i){
        float diff_x = vec1[i].x - vec2[i].x;
        float diff_y = (vec1[i].y - vec2[i].y)*camera_fx/camera_fy;
        // float diff_y = (vec1[i].y - vec2[i].y);
        result.emplace_back(diff_x,diff_y);
    }
    return result;
}

std::vector<cv::Point2f> OFCal(const std::vector<cv::Point2f> &pixels,const cv::Mat &frame1, const cv::Mat &frame2,std::vector<uchar> &status,float camera_fx = 1.0F,float camera_fy = 1.0F)
{
    int maxlevel = 4;
    std::vector<float> error;
    std::vector<cv::Point2f> opticalFlow;
    opticalFlow.reserve(pixels.size());
    cv::calcOpticalFlowPyrLK(frame1,frame2,pixels,opticalFlow,status, error,cv::Size(21,21),maxlevel);
    return diff(opticalFlow,pixels,camera_fx,camera_fy);
}

float WeightSum(const std::vector<cv::Point2f> &neighbours ,const int pos, const std::vector<float> &weight,int locate)
{

    float sum = 0;
    for(size_t i = 0; i < 8; i ++){
        if(locate == 0){
            sum += neighbours[i+pos].x * weight[i];
        }
        else if(locate == 1)
        {
            sum += neighbours[i+pos].y * weight[i];
        }
    }
    return sum;
}

float GraudOp(const std::vector<cv::Point2f> &neighbors, const int pos)
{
    float Graud = 1;
    Graud = WeightSum(neighbors,pos,f_x,1) - WeightSum(neighbors,pos,f_y,0);
    return Graud;
}

bool WhetherChase(const std::vector<cv::Point2f> &neighbors,const std::vector<uchar> & status, const int pos)
{
    for( size_t i =0;i < 8;i++){
        if(status[i+pos]==0){
            return false;
        }
    }
    return true;
}

std::vector<cv::Point3f> Pixel23D(std::vector<cv::Point2f> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2,float camera_fx = 1.0F,float camera_fy = 1.0F)
{
    std::vector<cv::Point3f> KP3D;
    std::vector<cv::Point2f> pix_surround;
    std::vector<cv::Point2f> OFuCD;
    std::vector<uchar> status;
    float tmp = 0;
    pix_surround.reserve(8*keypoints_1.size());
    OFuCD.reserve(keypoints_1.size());
    KP3D.reserve(keypoints_1.size());
    for (const auto& point2d : keypoints_1){
        pix_surround = Get8Pix(point2d,pix_surround);
    }
    OFuCD = OFCal(pix_surround,img1,img2,status,camera_fx,camera_fy);
    for (int i =0; i <OFuCD.size(); i = i+8 ){
        if(WhetherChase(OFuCD,status,i))
        {
            tmp = GraudOp(OFuCD,i);
            KP3D.emplace_back(keypoints_1[i/8].x,keypoints_1[i/8].y,tmp);
        }
    }
    return KP3D;
}

double point2PlaneDistance(const cv::Point3f& point, const PlaneModel& plane)
{
    return std::abs(static_cast<double>(plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d)) /
           std::sqrt(static_cast<double> (plane.a * plane.a + plane.b * plane.b + plane.c * plane.c));
}

std::vector<cv::Point2f> estimatePlaneRansac(const std::vector<cv::Point3f> &points,int  max_iterations,double threshold)
{
    // PlaneModel best_plane;
    std::vector<cv::Point2f> static_kps;
    int num_points = points.size();
    int num_best_inliers = 0;
    cv::RNG rng;
    for(int iteration = 0; iteration < max_iterations; iteration ++){
        int index1 = rng.uniform(0,num_points);
        int index2 = rng.uniform(0,num_points);
        int index3 = rng.uniform(0,num_points);
        if( index1 == index2 || index1 == index3 || index2 == index3){
            continue;
        }
        cv::Point3f p1 = points[index1];
        cv::Point3f p2 = points[index2];
        cv::Point3f p3 = points[index3];
        cv::Point3f normal = (p2 - p1).cross(p3 - p2);
        double norm = cv::norm(normal);
        if(norm < std::numeric_limits<double>::epsilon()){
            continue;
        }
        normal /= norm;
        PlaneModel plane;
        plane.a = normal.x;
        plane.b = normal.y;
        plane.c = normal.z;
        plane.d = -(plane.a * p1.x + plane.b * p1.y + plane.c * p1.z);
        std::vector<cv::Point2f> current_inliers;
        for (int i = 0; i < num_points; ++i) {
            double distance = point2PlaneDistance(points[i], plane);
            if (distance < threshold) {
                current_inliers.emplace_back(points[i].x,points[i].y);
            }
        }
        if(current_inliers.size() > num_best_inliers){
            num_best_inliers = current_inliers.size();
            static_kps = current_inliers;
            cv::Point3f new_normal(0, 0, 0);
            for (const auto& inlier : current_inliers) {
                new_normal += cv::Point3f(inlier.x, inlier.y, 0);
            }
            new_normal = new_normal /static_cast<float>(current_inliers.size());
            new_normal /= cv::norm(new_normal);
            plane.a = new_normal.x;
            plane.b = new_normal.y;
            plane.c = new_normal.z;
            plane.d = -(plane.a * p1.x + plane.b * p1.y + plane.c * p1.z);
        }
    }
    return static_kps;
}

//outlier version
void estimatePlaneRansac(const std::vector<cv::Point3f> &points, std::vector<cv::Point2f> &static_kps,std::vector<cv::Point2f> &dynamic_kps,int  max_iterations,double threshold)
{
    // PlaneModel best_plane;
    int num_points = points.size();
    int num_best_inliers = 0;
    cv::RNG rng;
    for(int iteration = 0; iteration < max_iterations; iteration ++){
        int index1 = rng.uniform(0,num_points);
        int index2 = rng.uniform(0,num_points);
        int index3 = rng.uniform(0,num_points);
        if( index1 == index2 || index1 == index3 || index2 == index3){
            continue;
        }
        cv::Point3f p1 = points[index1];
        cv::Point3f p2 = points[index2];
        cv::Point3f p3 = points[index3];
        cv::Point3f normal = (p2 - p1).cross(p3 - p2);
        double norm = cv::norm(normal);
        if(norm < std::numeric_limits<double>::epsilon()){
            continue;
        }
        normal /= norm;
        PlaneModel plane;
        plane.a = normal.x;
        plane.b = normal.y;
        plane.c = normal.z;
        plane.d = -(plane.a * p1.x + plane.b * p1.y + plane.c * p1.z);
        std::vector<cv::Point2f> current_inliers;
        std::vector<cv::Point2f> current_outliers;
        for (int i = 0; i < num_points; ++i) {
            double distance = point2PlaneDistance(points[i], plane);
            if (distance < threshold) {
                current_inliers.emplace_back(points[i].x,points[i].y);
            }
            else {
                current_outliers.emplace_back(points[i].x,points[i].y);
            }
        }
        if(current_inliers.size() > num_best_inliers){
            num_best_inliers = current_inliers.size();
            static_kps = current_inliers;
            dynamic_kps = current_outliers;
            cv::Point3f new_normal(0, 0, 0);
            for (const auto& inlier : current_inliers) {
                new_normal += cv::Point3f(inlier.x, inlier.y, 0);
            }
            new_normal = new_normal /static_cast<float>(current_inliers.size());
            new_normal /= cv::norm(new_normal);
            plane.a = new_normal.x;
            plane.b = new_normal.y;
            plane.c = new_normal.z;
            plane.d = -(plane.a * p1.x + plane.b * p1.y + plane.c * p1.z);
        }
    }
}

void KeypointDynamicOpticalDef(std::vector<cv::Point2f> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2,std::vector<cv::Point2f> &static_kps, std::vector<cv::Point2f> &dynamic_kps, double threshold,float camera_fx = 1.0F,float camera_fy = 1.0F)
{
    std::vector<cv::Point3f> p_uv3D;
    p_uv3D = Pixel23D(keypoints_1,img1,img2,camera_fx,camera_fy);
    PlaneModel best_plane;//ax+by+cz+d = 0
    estimatePlaneRansac(p_uv3D,static_kps,dynamic_kps,3000,threshold);
}

std::vector<cv::Point2f> OpticalDefStatic(std::vector<cv::Point2f> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2, double threshold,float camera_fx = 1.0F,float camera_fy = 1.0F)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    std::vector<cv::Point3f> p_uv3D;
    p_uv3D.reserve(keypoints_1.size());
    p_uv3D = Pixel23D(keypoints_1,img1,img2,camera_fx,camera_fy);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout<<"Optical feature detetction:"<<time_used.count()<<std::endl;
    t1 = std::chrono::steady_clock::now();
    PlaneModel best_plane;//ax+by+cz+d = 0
    std::vector<cv::Point2f> static_kps;
    static_kps.reserve(p_uv3D.size());
    static_kps = estimatePlaneRansac(p_uv3D,3000,threshold);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout<<"Optical plane estimation:"<<time_used.count()<<std::endl;
    return static_kps;
    // cv::fitPlane(p_uv3D, plane_model, cv::RANSAC, 0.01);
}

void DisplayDynamic(std::vector<cv::Point2f> &Dynamic_kps,std::vector<cv::Point2f> &Static_kps,cv::Mat &img1)
{
    for (const auto& kp : Static_kps){
        cv::circle(img1,kp,5,cv::Scalar(0,255,0),-1);
    }
    for(const auto& kp : Dynamic_kps){
        cv::circle(img1,kp,5,cv::Scalar(0,0,255),-1);
    }

    cv::imshow("feature Points",img1);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void savePicture(std::vector<cv::Point2f> &Dynamic_kps,std::vector<cv::Point2f> &Static_kps,cv::Mat &img1,const std::string &path)
{
    for (const auto& kp : Static_kps){
        cv::circle(img1,kp,5,cv::Scalar(0,255,0),-1);
    }
    for(const auto& kp : Dynamic_kps){
        cv::circle(img1,kp,5,cv::Scalar(0,0,255),-1);
    }
    cv::imwrite(path,img1);
}

std::vector<float> Pixel23D(std::vector<cv::KeyPoint> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2,float camera_fx = 1.0F,float camera_fy = 1.0F)
{
    std::vector<float> z;
    std::vector<uchar> status;
    std::vector<cv::Point2f> OFuCD;
    std::vector<cv::Point2f> pix_surround;
    pix_surround.reserve(8*keypoints_1.size());
    z.reserve(keypoints_1.size());
    OFuCD.reserve(keypoints_1.size());
    float tmp = 0;
    for(const auto& point2d : keypoints_1){
        pix_surround = Get8Pix(point2d.pt,pix_surround);
    }
    OFuCD = OFCal(pix_surround,img1,img2,status,camera_fx,camera_fy);
    for(int i = pix_surround.size() ; i > 0 ; i = i-8){
        if(WhetherChase(OFuCD,status,i-8))
        {
            tmp = GraudOp(OFuCD,i-8);
            z.push_back(tmp);
        }
        else
        {
            keypoints_1.erase(keypoints_1.begin()+(i-8)/8);
        }
    }
    std::reverse(z.begin(),z.end());
    return z;
}

double point2PlaneDistance(const cv::Point2f& pointxy, float kpz, PlaneModel& plane)
{
    return std::abs(static_cast<double>(plane.a * pointxy.x + plane.b * pointxy.y + plane.c * kpz + plane.d)) /
           std::sqrt(static_cast<double> (plane.a * plane.a + plane.b * plane.b + plane.c * plane.c));
}

void estimatePlaneRansac(const std::vector<cv::KeyPoint> &pointxy,const std::vector<float> &kpz, std::vector<cv::KeyPoint> &static_kps,std::vector<cv::KeyPoint> &dynamic_kps,int  max_iterations,double threshold)
{
    // PlaneModel best_plane;
    int num_points = pointxy.size();
    int num_best_inliers = 0;
    cv::RNG rng;
    for(int iteration = 0; iteration < max_iterations; iteration ++){
        int index1 = rng.uniform(0,num_points);
        int index2 = rng.uniform(0,num_points);
        int index3 = rng.uniform(0,num_points);
        if( index1 == index2 || index1 == index3 || index2 == index3){
            continue;
        }
        cv::Point3f p1 = cv::Point3f(pointxy[index1].pt.x,pointxy[index1].pt.y,kpz[index1]);
        cv::Point3f p2 = cv::Point3f(pointxy[index2].pt.x,pointxy[index2].pt.y,kpz[index2]);
        cv::Point3f p3 = cv::Point3f(pointxy[index3].pt.x,pointxy[index3].pt.y,kpz[index3]);
        cv::Point3f normal = (p2 - p1).cross(p3 - p2);
        double norm = cv::norm(normal);
        if(norm < std::numeric_limits<double>::epsilon()){
            continue;
        }
        normal /= norm;
        PlaneModel plane;
        plane.a = normal.x;
        plane.b = normal.y;
        plane.c = normal.z;
        plane.d = -(plane.a * p1.x + plane.b * p1.y + plane.c * p1.z);
        std::vector<cv::KeyPoint> current_inliers;
        std::vector<cv::KeyPoint> current_outliers;
        for (int i = 0; i < num_points; ++i) {
            double distance = point2PlaneDistance(pointxy[i].pt,kpz[i], plane);
            if (distance < threshold) {
                current_inliers.push_back(pointxy[i]);
            }
            else {
                current_outliers.push_back(pointxy[i]);
            }
        }
        if(current_inliers.size() > num_best_inliers){
            num_best_inliers = current_inliers.size();
            static_kps = current_inliers;
            dynamic_kps = current_outliers;
            cv::Point3f new_normal(0, 0, 0);
            for (const auto& inlier : current_inliers) {
                new_normal += cv::Point3f(inlier.pt.x, inlier.pt.y, 0);
            }
            new_normal = new_normal /static_cast<float>(current_inliers.size());
            new_normal /= cv::norm(new_normal);
            plane.a = new_normal.x;
            plane.b = new_normal.y;
            plane.c = new_normal.z;
            plane.d = -(plane.a * p1.x + plane.b * p1.y + plane.c * p1.z);
        }
    }
}

std::vector<cv::KeyPoint> estimatePlaneRansac(const std::vector<cv::KeyPoint> &pointxy,std::vector<float> &kps_z,int  max_iterations,double threshold)
{
    // PlaneModel best_plane;
    std::vector<cv::KeyPoint> static_kps;
    int num_points = pointxy.size();
    int num_best_inliers = 0;
    cv::RNG rng;
    for(int iteration = 0; iteration < max_iterations; iteration ++){
        int index1 = rng.uniform(0,num_points);
        int index2 = rng.uniform(0,num_points);
        int index3 = rng.uniform(0,num_points);
        if( index1 == index2 || index1 == index3 || index2 == index3){
            continue;
        }
        cv::Point3f p1 = cv::Point3f(pointxy[index1].pt.x,pointxy[index1].pt.y,kps_z[index1]);
        cv::Point3f p2 = cv::Point3f(pointxy[index2].pt.x,pointxy[index2].pt.y,kps_z[index2]);
        cv::Point3f p3 = cv::Point3f(pointxy[index3].pt.x,pointxy[index3].pt.y,kps_z[index3]);
        cv::Point3f normal = (p2 - p1).cross(p3 - p2);
        double norm = cv::norm(normal);
        if(norm < std::numeric_limits<double>::epsilon()){
            continue;
        }
        normal /= norm;
        PlaneModel plane;
        plane.a = normal.x;
        plane.b = normal.y;
        plane.c = normal.z;
        plane.d = -(plane.a * p1.x + plane.b * p1.y + plane.c * p1.z);
        std::vector<cv::KeyPoint> current_inliers;
        for (int i = 0; i < num_points; ++i) {
            double distance = point2PlaneDistance(pointxy[i].pt,kps_z[i], plane);
            if (distance < threshold) {
                current_inliers.push_back(pointxy[i]);
            }
        }
        if(current_inliers.size() > num_best_inliers){
            num_best_inliers = current_inliers.size();
            static_kps = current_inliers;
            cv::Point3f new_normal(0, 0, 0);
            for (const auto& inlier : current_inliers) {
                new_normal += cv::Point3f(inlier.pt.x, inlier.pt.y, 0);
            }
            new_normal = new_normal /static_cast<float>(current_inliers.size());
            new_normal /= cv::norm(new_normal);
            plane.a = new_normal.x;
            plane.b = new_normal.y;
            plane.c = new_normal.z;
            plane.d = -(plane.a * p1.x + plane.b * p1.y + plane.c * p1.z);
        }
    }
    return static_kps;
}

void KeypointDynamicOpticalDef(std::vector<cv::KeyPoint> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2,std::vector<cv::KeyPoint> &static_kps, std::vector<cv::KeyPoint> &dynamic_kps, double threshold,float camera_fx = 1.0F,float camera_fy = 1.0F)
{
    std::vector<float> kps_z;
    kps_z.reserve(keypoints_1.size());
    kps_z = Pixel23D(keypoints_1,img1,img2,camera_fx,camera_fy);
    estimatePlaneRansac(keypoints_1,kps_z,static_kps,dynamic_kps,3000,threshold);

}

std::vector<cv::KeyPoint> OpticalDefStatic(std::vector<cv::KeyPoint> &keypoints_1,const cv::Mat &img1,const cv::Mat &img2, double threshold,float camera_fx = 1.0F,float camera_fy = 1.0F)
{
    std::vector<float> kps_z;
    kps_z.reserve(keypoints_1.size());
    kps_z = Pixel23D(keypoints_1,img1,img2,camera_fx,camera_fy);
    std::vector<cv::KeyPoint> static_kps;
    static_kps = estimatePlaneRansac(keypoints_1,kps_z,3000,threshold);
    return static_kps;
}

void DisplayDynamic(std::vector<cv::KeyPoint> &Dynamic_kps,std::vector<cv::KeyPoint> &Static_kps,cv::Mat &img1)
{
    for (const auto& kp : Static_kps){
        cv::circle(img1,kp.pt,5,cv::Scalar(0,255,0),-1);
    }
    for(const auto& kp : Dynamic_kps){
        cv::circle(img1,kp.pt,5,cv::Scalar(0,0,255),-1);
    }

    cv::imshow("feature Points",img1);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void savePicture(std::vector<cv::KeyPoint> &Dynamic_kps,std::vector<cv::KeyPoint> &Static_kps,cv::Mat &img1,const std::string &path)
{
    for (const auto& kp : Static_kps){
        cv::circle(img1,kp.pt,5,cv::Scalar(0,255,0),-1);
    }
    for(const auto& kp : Dynamic_kps){
        cv::circle(img1,kp.pt,5,cv::Scalar(0,0,255),-1);
    }
    cv::imwrite(path,img1);
}
