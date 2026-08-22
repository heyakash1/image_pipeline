#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<uint8_t> invert(py::array_t<uint8_t> input){
    // body goes here
    py::buffer_info buf = input.request();

    uint8_t* pixels = reinterpret_cast<uint8_t*>(buf.ptr);
    int height = buf.shape[0];
    int width = buf.shape[1];

    py::array_t<uint8_t> result({height,width});

    uint8_t* outPixels = reinterpret_cast<uint8_t*>(result.request().ptr);

    long long iterations = height*width;
    for(long long i=0;i<iterations;i++){
        outPixels[i] = 255-pixels[i];
    }

    return result;
}

PYBIND11_MODULE(image_pipeline_cpp, m) {
    m.def("invert", &invert, "Inverts a grayscale image");
}