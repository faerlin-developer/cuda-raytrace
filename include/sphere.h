#ifndef SPHERE_H
#define SPHERE_H

#include "cuda.h"

struct Coord {
    float x;
    float y;
    float z;
};

struct Color {
    float r;
    float g;
    float b;
};

class Sphere {

public:
    float radius;
    Coord coord;
    Color color;

public:

    __device__ bool hit(float ox, float oy, float *dz);

};

#endif
