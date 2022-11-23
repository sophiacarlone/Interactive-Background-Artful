#include "object_tracking.h"

using namespace std;
using namespace cv;

const bool DETECT_OBJECT = false;

Object_Track::Object_Track(){
    //POSSIBLY: move this down to setObjectHSV
    //to detect a new object, set the lows to 0 and highs to 255		 
    Hue_Low_  = 0;//lower range of hue//
    Hue_high_ = 255;//upper range of hue//
    Sat_Low_  = 0;//lower range of saturation//
    Sat_high_ = 255;//upper range of saturation//
    Val_Low_  = 0;//lower range of value//
    Val_high_ = 255;//upper range of value//
    posX_ = 0.0;
    posY_ = 0.0;
    velocityX_ = 0.0;
    velocityY_ = 0.0;
    // engine::Engine engine_; //doesnt seem like this needs to be here

}

void Object_Track::setObjectHSV(){
    if (!DETECT_OBJECT) {
       Hue_Low_  = 36;//lower range of hue//
       Hue_high_ = 91;//upper range of hue//
       Sat_Low_  = 51;//lower range of saturation//
       Sat_high_ = 255;//upper range of saturation//
       Val_Low_  = 83;//lower range of value//
       Val_high- = 166;//upper range of value//
   }
   
   /*USE FOR OBJECT DETECTING INFORMATION*/
//    createTrackbar("LowH", "Adjust", &Hue_Low, 179);//track-bar for min hue//
//    createTrackbar("HighH","Adjust", &Hue_high, 179);//track-bar for max hue//
//    createTrackbar("LowS", "Adjust", &Sat_Low, 255);//track-bar for min saturation//
//    createTrackbar("HighS", "Adjust", &Sat_high, 255);// track-bar for max saturation//
//    createTrackbar("LowV", "Adjust", &Val_Low,255);//track-bar for min value//
//    createTrackbar("HighV", "Adjust", &Val_high, 255);// track - bar for max value//  

}

void Object_Track::run(){
    // @TODO organize the computer vision stuff into a separate object or something
    int vidnum;
    cout << "Camera number: ";
    cin >> vidnum;
	//TODO: have openCV find best camera
   
    VideoCapture video_load(vidnum);//capturing video from default camera//
    double fps = video_load.get(CAP_PROP_FPS); //fps for velocity calculations
    namedWindow("Adjust");//declaring window to show the image//

    setObjectHSV();

    float horizontal_Last = -1;//initial horizontal position//
    float vertical_Last = -1;//initial vertical position//

    Mat temp;//declaring a matrix to load frames from video stream//
    video_load.read(temp);//loading frames from video stream//

    int camera_size_vertical = temp.rows;
    int camera_size_horizontal = temp.cols;

    Mat sprite = imread("./sprites/1.jpg");

    //    float posX, posY;
    //    float velocityX, velocityY;
    //    engine::Engine engine;

//TODO: Main should call engine.run();
    engine.run([&] () {
        Mat actual_Image;//declaring a matrix for actual image//
        bool temp_load = video_load.read(actual_Image);//loading frames from video to the matrix//
        flip(actual_Image, actual_Image, 1); // mirror so it's more intuitive for user
        Mat converted_to_HSV;//declaring a matrix to store converted image//
        cvtColor(actual_Image, converted_to_HSV, COLOR_BGR2HSV);//converting BGR image to HSV//
        Mat adjusted_frame;//declaring a matrix to detected color//

        inRange(converted_to_HSV,Scalar(Hue_Low_, Sat_Low_, Val_Low_),
        Scalar(Hue_high_, Sat_high_, Val_high_), adjusted_frame);//applying change of values of track-bars//        
        // get rid of tiny white pixels due to noise
        erode(adjusted_frame,adjusted_frame,getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        dilate(adjusted_frame, adjusted_frame,getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        // get rid of tiny holes in white objects due to noise
        dilate(adjusted_frame, adjusted_frame,getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        erode(adjusted_frame, adjusted_frame, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        
        Moments detecting_object = moments(adjusted_frame);//creating an object from detected color frame//
        double vertical_moment = detecting_object.m01;//getting value of vertical position//
        double horizontal_moment = detecting_object.m10;//getting value of horizontal position//
        double tracking_area = detecting_object.m00;//getting area of the object//

        if (tracking_area > 10000){ //when area of the object is greater than 10000 pixels//
            posX_ = ((horizontal_moment / tracking_area) / camera_size_horizontal) * 2 - 1; //calculate the horizontal position of the object//
            posY_ = ((vertical_moment / tracking_area) / camera_size_vertical) * 2 - 1; //calculate the vertical position of the object//

            velocityX_ = (posX_ - horizontal_Last)/(1./fps);
            velocityY_ = (posY_ - vertical_Last)/(1./fps);
            
            horizontal_Last = posX_;//getting new horizontal position//
            vertical_Last = posY_;// getting new vertical position value//

            engine.update_position(glm::vec2(posX_, posY_));
        }
        
        imshow("Actual", actual_Image);//showing original video//
        
        // @TODO why are we waiting so long? this might significantly affect framerate
        if(waitKey(10)==27){ //if esc is pressed loop will break//
            engine.stop(); //exit(0); // @TODO replace with `engine.stop()` or something
        }      
    });
}