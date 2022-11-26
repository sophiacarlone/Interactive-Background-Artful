#include "tracker.h"

using namespace std;
using namespace cv;
using namespace tracker;

const bool DETECT_OBJECT = false;

Tracker::Tracker() {
    //POSSIBLY: move this down to setObjectHSV
    //to detect a new object, set the lows to 0 and highs to 255		 
    Hue_Low_  = 0;//lower range of hue//
    Hue_high_ = 255;//upper range of hue//
    Sat_Low_  = 0;//lower range of saturation//
    Sat_high_ = 255;//upper range of saturation//
    Val_Low_  = 0;//lower range of value//
    Val_high_ = 255;//upper range of value//
    
    horizontal_Last_ = -1;//initial horizontal position//
    vertical_Last_ = -1;//initial vertical position//

    posX_ = 0.0;
    posY_ = 0.0;
    // velocityX_ = 0.0;
    // velocityY_ = 0.0;
}

//TODO: can change parameters to detect other object. Wanted to keep defaults and specifics separate for now
void Tracker::setObjectHSV(){
    if (!DETECT_OBJECT) {
        Hue_Low_  = 36;//lower range of hue//
        Hue_high_ = 91;//upper range of hue//
        Sat_Low_  = 51;//lower range of saturation//
        Sat_high_ = 255;//upper range of saturation//
        Val_Low_  = 83;//lower range of value//
        Val_high_ = 166;//upper range of value//
   }
   
	/*USE FOR OBJECT DETECTING INFORMATION*/
	//    createTrackbar("LowH", "Adjust", &Hue_Low, 179);//track-bar for min hue//
	//    createTrackbar("HighH","Adjust", &Hue_high, 179);//track-bar for max hue//
	//    createTrackbar("LowS", "Adjust", &Sat_Low, 255);//track-bar for min saturation//
	//    createTrackbar("HighS", "Adjust", &Sat_high, 255);// track-bar for max saturation//
	//    createTrackbar("LowV", "Adjust", &Val_Low,255);//track-bar for min value//
	//    createTrackbar("HighV", "Adjust", &Val_high, 255);// track - bar for max value//  

}

void Tracker::run(int vidnum, bool show_windows) {
	if (show_windows) {
		namedWindow("Actual");
		namedWindow("Detected_Object");
	}

	VideoCapture vid(vidnum);
	int camera_size_vertical   = (int)vid.get(CAP_PROP_FRAME_HEIGHT);
	int camera_size_horizontal = (int)vid.get(CAP_PROP_FRAME_WIDTH );

	Mat input_image;
	Mat converted_to_HSV;
	Mat adjusted_frame;

	while (true) {
		// double fps = video_load.get(CAP_PROP_FPS); //fps for velocity calculations
		
		bool temp_load = vid.read(input_image);//loading frames from video to the matrix//
		flip(input_image, input_image, 1); // mirror so it's more intuitive for user
		cvtColor(input_image, converted_to_HSV, COLOR_BGR2HSV);//converting BGR image to HSV//
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
			posY_ = ((vertical_moment / tracking_area) / camera_size_vertical) * 2 - 1; //calculate the vertical position of the object// // if (horizontal_Last >= 0 && vertical_Last >= 0 && posX >= 0 && posY >= 0){ //when the detected object moves//
			//    line(track_motion, Point(posX, posY), Point(horizontal_Last, vertical_Last), Scalar(0, 0, 255), 2);//draw lines of red color on the path of detected object;s motion//
			// }

		//  //? do we still use the object velocity here
		//  object.setVelocityX(horizontal_Last, fps);
		//  object.setVelocityY(vertical_Last, fps);
			
			horizontal_Last_ = posX_;//getting new horizontal position//
			vertical_Last_ = posY_;// getting new vertical position value//
		}
		
		if (show_windows) {
			imshow("Actual", input_image);//showing original video//
			imshow("Detected_Object", adjusted_frame);//showing detected object//
			// actual_Image = actual_Image + track_motion;//drawing continuous line in original video frames//
		}
		waitKey(1); // waitKey() is required for `imshow` to actually show something
	}
}
