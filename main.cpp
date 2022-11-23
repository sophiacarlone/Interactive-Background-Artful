#include <iostream>
#include "include/object_tracking.h"
#include <ostream>
#include <glm/glm.hpp>
#include "include/engine.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
   // @TODO organize the computer vision stuff into a separate object or something
    int vidnum;
    cout << "Camera number: ";
    cin >> vidnum;
	//TODO: have openCV find best camera
   
   VideoCapture video_load(vidnum);//capturing video from default camera//
   double fps = video_load.get(CAP_PROP_FPS); //fps for velocity calculations
   namedWindow("Adjust");//declaring window to show the image//

   float horizontal_Last = -1;//initial horizontal position//
   float vertical_Last = -1;//initial vertical position//
   
   Mat temp;//declaring a matrix to load frames from video stream//
   video_load.read(temp);//loading frames from video stream//
   
   // Mat track_motion = Mat::zeros(temp.size(), CV_8UC3);//creating black matrix for detection//
   
   int camera_size_vertical = temp.rows;
   int camera_size_horizontal = temp.cols;

   engine::Engine engine;

   object::Object_Track object(); //object tracked
   object.setObjectHSV();

   engine.run([&] () {
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
         object.setPosX(((horizontal_moment / tracking_area) / camera_size_horizontal) * 2 - 1); //calculate the horizontal position of the object//
         object.setPosY(((vertical_moment / tracking_area) / camera_size_vertical) * 2 - 1); //calculate the vertical position of the object//
         // if (horizontal_Last >= 0 && vertical_Last >= 0 && posX >= 0 && posY >= 0){ //when the detected object moves//
         //    line(track_motion, Point(posX, posY), Point(horizontal_Last, vertical_Last), Scalar(0, 0, 255), 2);//draw lines of red color on the path of detected object;s motion//
         // }

         //? do we still use the object velocity here
         object.setVelocityX(horizontal_Last, fps);
         object.setVelocityY(vertical_Last, fps);
         
         horizontal_Last = object.getPosX();//getting new horizontal position//
         vertical_Last = object.getPosY();// getting new vertical position value//

         engine.update_position(glm::vec2(object.getPosX(), object.getPosY()));
      }
      
      // imshow("Detected_Object", adjusted_frame);//showing detected object//
      // actual_Image = actual_Image + track_motion;//drawing continuous line in original video frames//
      imshow("Actual", actual_Image);//showing original video//
      
      // @TODO why are we waiting so long? this might significantly affect framerate
      if(waitKey(30)==27){ //if esc is pressed loop will break//
         engine.stop(); //exit(0); // @TODO replace with `engine.stop()` or something
      }      
   });
   return 0;
}
