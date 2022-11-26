#ifndef TRACKER_H
#define TRACKER_H

#include <bits/types/time_t.h>
#include <cstddef>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>

namespace tracker {

class Tracker{
    private:
        cv::VideoCapture vid_;
        bool show_windows_;
        int camera_size_vertical_;
        int camera_size_horizontal_;
        int Hue_Low_;//lower range of hue//
        int Hue_high_;//upper range of hue//
        int Sat_Low_;//lower range of saturation//
        int Sat_high_;//upper range of saturation//
        int Val_Low_;//lower range of value//
        int Val_high_;//upper range of value//
        float horizontal_Last_ = -1;//initial horizontal position//
        float vertical_Last_ = -1;//initial vertical position//
        float posX_, posY_;
        // float velocityX_, velocityY_;

        // storing some intermediate variables here to avoid reallocating every time getPos is called
        cv::UMat input_image_; // the image we get from the camera
        cv::UMat converted_to_HSV_;
        cv::UMat adjusted_frame_; // image after applying color thresholding
        
    public:
        Tracker(int vidnum, bool showWindows);
        void setObjectHSV();

        cv::Point2d getPos();
        // float getVelocityX()  {return velocityX_;}
        // float getVelocityY()  {return velocityY_;}

        // void setPosX(float x){posX_ = x;}
        // void setPosY(float y){posY_ = y;}
        
        // void setVelocityX(float horizontal_last, double fps){
        //     velocityX_ = (posX_ - horizontal_last)/(1./fps);
        // }
        // void setVelocityY(float vertical_Last, double fps){
        //     velocityY_ = (posY_ - vertical_Last)/(1./fps);
        // }

};

} //namespace tracker

#endif // TRACKER_H