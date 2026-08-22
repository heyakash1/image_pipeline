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


py::array_t<uint8_t> to_grayscale(py::array_t<uint8_t>input){
    py::buffer_info buf = input.request();

    uint8_t* pixels = reinterpret_cast<uint8_t*>(buf.ptr);
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = buf.shape[2];

    py::array_t<uint8_t> result({height,width});
    uint8_t*outPixels = reinterpret_cast<uint8_t*>(result.request().ptr);
    
    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            int r = pixels[(row*width + col) * channels + 0];
            int g = pixels[(row*width + col) * channels + 1];
            int b = pixels[(row*width + col) * channels + 2];

            int sum = r + g + b;
            int avg = sum/channels;
            uint8_t finalAvg = (uint8_t)avg;
            outPixels[row*width + col] = finalAvg;
        }
    }
    return result;
}
PYBIND11_MODULE(image_pipeline_cpp, m) {
    m.def("invert", &invert, "Inverts a grayscale image");
    m.def("to_grayscale", &to_grayscale, "Converts an RGB image to grayscale");
}