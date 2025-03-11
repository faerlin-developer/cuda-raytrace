
#include <cstdio>
#include "cuda.h"
#include "sphere.h"
#include "raytrace.h"
#include "args.h"

__constant__ Sphere d_spheres[NUM_SPHERES];

__global__ void raytrace_kernel(unsigned char *pixels) {

    auto x = threadIdx.x + blockIdx.x * blockDim.x;
    auto y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= LENGTH || y >= LENGTH) {
        return;
    }

    auto offset = x + y * blockDim.x * gridDim.x;

    float ox = (x - LENGTH / 2.0f);
    float oy = (y - LENGTH / 2.0f);

    float r = 0;
    float g = 0;
    float b = 0;
    float max_height = INT_MIN;
    for (int i = 0; i < NUM_SPHERES; i++) {
        float dz;
        auto isHit = d_spheres[i].hit(ox, oy, &dz);
        auto height = dz + d_spheres[i].coord.z;

        if (isHit && height > max_height) {
            auto scale = dz / sqrtf(d_spheres[i].radius * d_spheres[i].radius);
            r = d_spheres[i].color.r * scale;
            g = d_spheres[i].color.g * scale;
            b = d_spheres[i].color.b * scale;
            max_height = height;
        }
        /*
        if (t > maxz) {
            float fscale = n;
            r = d_spheres[i].color.r * fscale;
            g = d_spheres[i].color.g * fscale;
            b = d_spheres[i].color.b * fscale;
            maxz = t;
        }
         */
    }

    pixels[offset * 3 + 0] = static_cast<unsigned char>(r);
    pixels[offset * 3 + 1] = static_cast<unsigned char>(g);
    pixels[offset * 3 + 2] = static_cast<unsigned char>(b);
}