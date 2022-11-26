#include <iostream>
#include "include/tracker.h"
#include <ostream>
#include <glm/glm.hpp>
#include "include/engine.h"

using namespace std;
using namespace cv;

#ifdef NDEBUG
    const bool SHOW_TRACKER_WINDOWS = false;
#else
    const bool SHOW_TRACKER_WINDOWS = true;
#endif

int main(int argc, char** argv) {
    int vidnum;
    cout << "Camera number: ";
    cin >> vidnum;
	//TODO: have openCV find best camera
   
    engine::Engine engine;

    tracker::Tracker tracker(vidnum, SHOW_TRACKER_WINDOWS); //object tracker
    tracker.setObjectHSV();

    engine.run([&] () {
        cv::Point2d pos = tracker.getPos();
        engine.update_position(glm::vec2(pos.x, pos.y));
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