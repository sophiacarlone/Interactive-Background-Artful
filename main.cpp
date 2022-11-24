#include <iostream>
#include "include/tracker.h"
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


   Tracker tracker; //object tracked
   tracker.setObjectHSV();

   engine.run([&] () {
      tracker.run(vidnum);
      engine.update_position(glm::vec2(tracker.getPosX(), tracker.getPosY()));
   });
   
   return 0;
}

// #include <include/engine.h>
// #include <include/tracker.h>
// // ...

// int main() {
//   Tracker tracker();
//   Engine engine();

//   engine.run([&]() {
//     vec2 pos = tracker.get_position();
//     engine.update_position(pos);
//   });
// }