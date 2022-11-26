# Interactive-Background-Artful

Final project for Interactive and Algorithmic Art class at Clarkson University

Two main parts, the Graphics Engine and the Object Detection.

Incorporates OpenCV

## Building

The following will build all targets.
```bash
mkdir -p build
cd build
cmake ..
make
```
If you only want to build a specific target, you could replace `make` with any of:
```bash
make test         # to test the tracker and boids engine together
make test-tracker # to test just the tracker
make test-boids   # to test just the boids
```