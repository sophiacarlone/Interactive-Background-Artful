#include <bits/types/time_t.h>
#include <cstddef>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ostream>
#include <glm/glm.hpp>
#include "include/engine.h"

using namespace std;
using namespace cv;

const bool DETECT_OBJECT = false;

int main(int argc, char** argv) {
    int vidnum;
    cout << "Camera number: ";
    cin >> vidnum;
	//have openCV find best camera
   VideoCapture video_load(vidnum);//capturing video from default camera//
   double fps = video_load.get(CAP_PROP_FPS); //fps for velocity calculations
   namedWindow("Adjust");//declaring window to show the image//

   //to detect a new object, set the lows to 0 and highs to 255		 
   int Hue_Low  = 0;//lower range of hue//
   int Hue_high = 255;//upper range of hue//
   int Sat_Low  = 0;//lower range of saturation//
   int Sat_high = 255;//upper range of saturation//
   int Val_Low  = 0;//lower range of value//
   int Val_high = 255;//upper range of value//
   if (!DETECT_OBJECT) {
       Hue_Low  = 36;//lower range of hue//
       Hue_high = 91;//upper range of hue//
       Sat_Low  = 51;//lower range of saturation//
       Sat_high = 255;//upper range of saturation//
       Val_Low  = 83;//lower range of value//
       Val_high = 166;//upper range of value//
   }
   
   /*USE FOR OBJECT DETECTING INFORMATION*/
   createTrackbar("LowH", "Adjust", &Hue_Low, 179);//track-bar for min hue//
   createTrackbar("HighH","Adjust", &Hue_high, 179);//track-bar for max hue//
   createTrackbar("LowS", "Adjust", &Sat_Low, 255);//track-bar for min saturation//
   createTrackbar("HighS", "Adjust", &Sat_high, 255);// track-bar for max saturation//
   createTrackbar("LowV", "Adjust", &Val_Low,255);//track-bar for min value//
   createTrackbar("HighV", "Adjust", &Val_high, 255);// track - bar for max value//  
   /**/

   float horizontal_Last = -1;//initial horizontal position//
   float vertical_Last = -1;//initial vertical position//
   
   Mat temp;//declaring a matrix to load frames from video stream//
   video_load.read(temp);//loading frames from video stream//
   
   Mat track_motion = Mat::zeros(temp.size(), CV_8UC3);//creating black matrix for detection//
   
   int camera_size_vertical = temp.rows;
   int camera_size_horizontal = temp.cols;
   //cout << camera_size_horizontal << " " << camera_size_vertical << endl;   

   Mat sprite = imread("./sprites/1.jpg");

   float posX, posY;

   float velocityX, velocityY;

   engine::Engine engine;

   engine.run([
      &engine,
      &posX, &posY, &velocityX, &velocityY,
      &temp,
      &horizontal_Last, &vertical_Last,
      &camera_size_vertical, &camera_size_horizontal,
      &fps,
      &video_load,
      &Hue_Low, &Hue_high, &Sat_Low, &Sat_high, &Val_Low, &Val_high,
      &track_motion
   ] () {
      Mat actual_Image;//declaring a matrix for actual image//
      bool temp_load = video_load.read(actual_Image);//loading frames from video to the matrix//
      flip(actual_Image, actual_Image, 1); // mirror so it's more intuitive for user
      Mat converted_to_HSV;//declaring a matrix to store converted image//
      cvtColor(actual_Image, converted_to_HSV, COLOR_BGR2HSV);//converting BGR image to HSV//
      Mat adjusted_frame;//declaring a matrix to detected color//

      inRange(converted_to_HSV,Scalar(Hue_Low, Sat_Low, Val_Low),
      Scalar(Hue_high, Sat_high, Val_high), adjusted_frame);//applying change of values of track-bars//        
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
         posX = ((horizontal_moment / tracking_area) / camera_size_horizontal) * 2 - 1; //calculate the horizontal position of the object//
         posY = ((vertical_moment / tracking_area) / camera_size_vertical) * 2 - 1; //calculate the vertical position of the object//
         // if (horizontal_Last >= 0 && vertical_Last >= 0 && posX >= 0 && posY >= 0){ //when the detected object moves//
         //    line(track_motion, Point(posX, posY), Point(horizontal_Last, vertical_Last), Scalar(0, 0, 255), 2);//draw lines of red color on the path of detected object;s motion//
         // }
	//cout << "posx: " << posX << " posy " << posY << endl;

         velocityX = (posX - horizontal_Last)/(1./fps);
         velocityY = (posY - vertical_Last)/(1./fps);
         
         horizontal_Last = posX;//getting new horizontal position//
         vertical_Last = posY;// getting new vertical position value//

         engine.update_position(glm::vec2(posX, posY));
      }
      
      imshow("Detected_Object", adjusted_frame);//showing detected object//
      actual_Image = actual_Image + track_motion;//drawing continuous line in original video frames//
      imshow("Actual", actual_Image);//showing original video//
      //cout << "position of the object is:" << Horizontal_Last << "," << vertical_Last << endl;//showing tracked co-ordinated values//
      //sprite.copyTo(actual_Image, (cv::Rect(posX,posY,sprite.cols, sprite.rows)));
      //cout << "XVELOCITY: " << velocityX << " YVELOCITY: " << velocityY << endl;
      
      if(waitKey(30)==27){ //if esc is pressed loop will break//
         //cout << "esc key is pressed by user" << endl;
         exit(0); // @todo replace with `engine.stop()` or something
      }
      
      //POTENTIAL RESET
      // else if (waitKey(30)==99){ //idea for reset by pressing 'c'. going to move on
      //    video_load.read(actual_Image);
      //    cout << "c was pressed" << endl;
      // } 
   });
   return 0;
}
