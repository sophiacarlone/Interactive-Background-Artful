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
   
   engine::Engine engine;

   Object_Track object = new Object_Track(); //object tracked
   object.setObjectHSV();

   engine.run([&] () {
      object.run(vidnum, engine);
   });
   
   return 0;
}
