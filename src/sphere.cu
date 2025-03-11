
#include "cuda.h"
#include "sphere.h"
#include <cstdio>

__device__ bool Sphere::hit(float ox, float oy, float *dz) {

    float dx = ox - coord.x;
    float dy = oy - coord.y;
    if (dx * dx + dy * dy < radius * radius) {
        *dz = sqrtf(radius * radius - dx * dx - dy * dy);
        return true;
    }

    return false;

}
