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
#include "include/engine.h"

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
    // engine::Engine engine_;
    //? So this object does nto need a engine object

    public:
    Object_Track();
    void run();
    void setObjectHSV();
    //TODO: see if these are useful anywhere
    float getPosX(){return posX_;}
    float getPosY(){return posY_;}
    float getVelocityX(){return velocityX_;}
    float getVelocityY(){return velocityY_;}
}

#endif // OBJECT_TRACKING_H