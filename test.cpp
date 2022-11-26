#include <iostream>
#include <thread>
#include "include/tracker.h"
#include "include/engine.h"

using std::cout, std::cin;
using engine::vec2;
using tracker::Point2d;

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
    tracker::Tracker tracker;
    tracker.setObjectHSV();

    // let tracker and engine run simultaneously
    std::thread engine_thread([&engine, &tracker]() {
        engine.run([&engine, &tracker]() {
            Point2d pos = tracker.getPos();
            engine.updateAttractor(vec2(pos.x, pos.y));
        });
    });
    // Note: keeping tracker in the main thread because it breaks when attempting to update its windows from
    // another thread (OpenCV limitation).
    tracker.run(vidnum, SHOW_TRACKER_WINDOWS);
    
    return 0;
}
