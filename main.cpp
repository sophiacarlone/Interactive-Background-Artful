#include <iostream>
#include "include/tracker.h"
#include <ostream>
#include <thread>
#include "include/engine.h"

using namespace std;
using namespace cv;
using engine::vec2;

#ifdef NDEBUG
    const bool SHOW_TRACKER_WINDOWS = false;
#else
    const bool SHOW_TRACKER_WINDOWS = true;
#endif

const size_t N_BOIDS = 30;

int main(int argc, char** argv) {
    int vidnum;
    cout << "Camera number: ";
    cin >> vidnum;
	//TODO: have openCV find best camera
   
    // initialize
    engine::Engine engine(N_BOIDS);
    tracker::Tracker tracker(vidnum, SHOW_TRACKER_WINDOWS);
    tracker.setObjectHSV();

    // let tracker and engine run simultaneously
    std::thread engine_thread([&engine, &tracker]() {
        engine.run([&engine, &tracker]() {
            cv::Point2d pos = tracker.getPos();
            engine.updateAttractor(vec2(pos.x, pos.y));
        });
    });
    // Note: keeping tracker in the main thread because it breaks when attempting to update its windows from
    // another thread (OpenCV limitation).
    tracker.run();
    
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