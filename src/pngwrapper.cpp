
#include <cstdio>
#include <vector>
#include "pngwrapper.h"

PngWrapper::PngWrapper(std::string filename, int height, int width) : height(height), width(width) {

    // Open output PNG file
    fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        fprintf(stderr, "cannot open file\n");
        return;
    }

    // Create PNG write structure
    png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        fprintf(stderr, "cannot create PNG write struct\n");
        fclose(fp);
        return;
    }

    // Create PNG info structure
    info = png_create_info_struct(png);
    if (!info) {
        fprintf(stderr, "cannot create PNG info struct\n");
        png_destroy_write_struct(&png, nullptr);
        fclose(fp);
        return;
    }

    png_init_io(png, fp);

    // Set info fields
    png_set_IHDR(
            png, info, width, height,
            8, PNG_COLOR_TYPE_RGB,
            PNG_INTERLACE_NONE,
            PNG_COMPRESSION_TYPE_DEFAULT,
            PNG_FILTER_TYPE_DEFAULT
    );

    // Initialize PNG file and write header
    png_write_info(png, info);
}

void PngWrapper::write(std::vector<unsigned char> &pixels) {

    std::vector<png_bytep> rows(height);
    for (int i = 0; i < height; i++) {
        rows[i] = &pixels[i * width * 3];
    }

    png_write_image(png, rows.data());
}

void PngWrapper::close() {
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
}
