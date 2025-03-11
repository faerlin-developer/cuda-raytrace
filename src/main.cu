#include <vector>
#include <random>
#include "cuda.h"
#include "pngwrapper.h"
#include "sphere.h"
#include "raytrace.h"
#include "call.h"
#include "args.h"

#include <iostream>

extern Sphere *d_spheres;

int main() {

    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Constant Memory Size: " << prop.totalConstMem << " bytes" << std::endl;

    unsigned char *d_pixels;
    CALL(cudaMalloc(&d_pixels, 3 * LENGTH * LENGTH * sizeof(unsigned char)));

    printf("d_spheres size: %ld\n", 20 * sizeof(Sphere));

    Sphere h_spheres[NUM_SPHERES];

    std::random_device rd;
    std::mt19937 gen(42); // Mersenne Twister engine
    std::uniform_real_distribution<float> rand_color(0, 255);
    std::uniform_real_distribution<float> rand_coord(-500, 500);
    std::uniform_real_distribution<float> rand_radius(20, 120);

    for (auto &h_sphere: h_spheres) {

        h_sphere.color.r = rand_color(gen);
        h_sphere.color.g = rand_color(gen);
        h_sphere.color.b = rand_color(gen);

        h_sphere.coord.x = rand_coord(gen);
        h_sphere.coord.y = rand_coord(gen);
        h_sphere.coord.z = rand_coord(gen);

        h_sphere.radius = rand_radius(gen);
    }

    // Copy to constant memory
    CALL(cudaMemcpyToSymbol(d_spheres, h_spheres, NUM_SPHERES * sizeof(Sphere)));

    auto block_dim = dim3(16, 16);
    auto grid_dim = dim3((LENGTH - 1) / block_dim.x + 1, (LENGTH - 1) / block_dim.y + 1);
    raytrace_kernel<<<grid_dim, block_dim>>>(d_pixels);
    CALL(cudaDeviceSynchronize());

    auto h_pixels = std::vector<unsigned char>(3 * LENGTH * LENGTH, 0);
    CALL(cudaMemcpy(h_pixels.data(), d_pixels, 3 * LENGTH * LENGTH * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    auto png = PngWrapper("output.png", LENGTH, LENGTH);
    png.write(h_pixels);
    png.close();

    cudaFree(d_pixels);

    return 0;
}
