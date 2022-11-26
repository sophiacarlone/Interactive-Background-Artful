#ifndef OBJECT_TRACKING_H
#define OBJECT_TRACKING_H

#include <bits/types/time_t.h>
#include <cstddef>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <glm/glm.hpp>

class Object_Track{
    public:
    int Hue_Low_;//lower range of hue//
    int Hue_high_;//upper range of hue//
    int Sat_Low_;//lower range of saturation//
    int Sat_high_;//upper range of saturation//
    int Val_Low_;//lower range of value//
    int Val_high_;//upper range of value//
    float posX_, posY_;
    float velocityX_, velocityY_;

    public:
    Object_Track();
    void run(int vidnum, Engine &engine);
    void setObjectHSV();

    float getPosX()       {return posX_;}
    float getPosY()       {return posY_;}
    // float getVelocityX()  {return velocityX_;}
    // float getVelocityY()  {return velocityY_;}

    void setPosX(float x){posX_ = x;}
    void setPosY(float y){posY_ = y;}
    
    void setVelocityX(float horizontal_last, double fps){
        velocityX_ = (posX_ - horizontal_last)/(1./fps);
    }
    void setVelocityY(float vertical_Last, double fps){
        velocityY_ = (posY_ - vertical_Last)/(1./fps);
    }

}

#endif // OBJECT_TRACKING_H