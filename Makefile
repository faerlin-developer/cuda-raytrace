
# The -rdc=true (relocatable device code) flag ensures that
# device functions across translation units can be resolved at link time.

build: bin
	nvcc -arch=sm_80 -rdc=true -o bin/main src/main.cu src/sphere.cu src/pngwrapper.cpp src/raytrace.cu -I include -lpng

run:
	bin/main

bin:
	rm -rf bin
	mkdir -p bin