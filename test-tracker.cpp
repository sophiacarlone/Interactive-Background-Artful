#include <iostream>
#include "include/tracker.h"

using std::cout, std::cin;

int main(int argc, char** argv) {
    int vidnum;
    cout << "Camera number: ";
    cin >> vidnum;
	//TODO: have openCV find best camera
   
    tracker::Tracker tracker;
    tracker.setObjectHSV();

    tracker.run(vidnum, true);
    
    return 0;
}
