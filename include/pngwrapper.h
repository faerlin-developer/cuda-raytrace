#ifndef PNGWRAPPER_H
#define PNGWRAPPER_H

#include <cstdio>
#include <png.h>
#include <string>

class PngWrapper {

private:
    FILE *fp;
    int height;
    int width;
    png_infop info;
    png_structp png;

public:

    PngWrapper(std::string filename, int height, int width);

    void write(std::vector<unsigned char> &pixels);

    void close();

};

#endif
