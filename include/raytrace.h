#ifndef RAYTRACE_H
#define RAYTRACE_H

#include "cuda.h"

__global__ void raytrace_kernel(unsigned char *pixels);

#endif
