#ifndef CALL_H
#define CALL_H

#include "cuda.h"

#define CALL(call) check_error((call), __FILE__, __LINE__)

inline void check_error(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "cuda runtime error at %s line %d: %s\n", file, line, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

#endif
