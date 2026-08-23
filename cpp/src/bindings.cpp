#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
namespace py = pybind11;

int clamp(int value, int lo, int hi){
    value = std::max(value,lo);
    value = std::min(value,hi);

    return value;
}

py::array_t<uint8_t> box_blur(py::array_t<uint8_t>input, int kernel_size){
    py::buffer_info buf = input.request();

    uint8_t* pixels = reinterpret_cast<uint8_t*>(buf.ptr);
    int height = buf.shape[0];
    int width = buf.shape[1];

    py::array_t<uint8_t>result({height,width});

    uint8_t*outPixels = reinterpret_cast<uint8_t*>(result.request().ptr);

    int half = kernel_size/2;
    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            int sum = 0;
            for(int dr= -half;dr<=half;dr++){
                for(int dc= -half;dc<=half;dc++){
                    int neighborRow = row+dr;
                    int neighborCol = col+dc;

                    neighborRow = clamp(neighborRow,0,height-1);
                    neighborCol = clamp(neighborCol,0,width-1);
                    int neighborVal = pixels[neighborRow*width+neighborCol];
                    sum += neighborVal;
                }
            }
            sum /= (kernel_size*kernel_size);
            outPixels[row*width+col] = sum;
        }
    }
    return result;
}
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

py::array_t<uint8_t> threshold(py::array_t<uint8_t> input, uint8_t cutoff){
    py::buffer_info buf = input.request();

    uint8_t* pixels = reinterpret_cast<uint8_t*>(buf.ptr);
    int height = buf.shape[0];
    int width = buf.shape[1];

    py::array_t<uint8_t>result({height,width});
    uint8_t*outPixels = reinterpret_cast<uint8_t*>(result.request().ptr);

    int iterations = height*width;
    for(int i = 0;i<iterations;i++){
        if(pixels[i] >= cutoff){
            outPixels[i] = 255;
        }
        else    outPixels[i] = 0;
    }
    return result;
}

py::array_t<uint8_t> gaussian_blur(py::array_t<uint8_t>input, int kernel_size, double sigma){
    py::buffer_info buf = input.request();

    uint8_t*pixels = reinterpret_cast<uint8_t*>(buf.ptr);
    int height = buf.shape[0];
    int width = buf.shape[1];

    py::array_t<uint8_t>result({height,width});
    uint8_t*outPixels = reinterpret_cast<uint8_t*>(result.request().ptr);

    int half = kernel_size/2;
    std::vector<double>weights(kernel_size*kernel_size);
    for(int dr=-half;dr<=half;dr++){
        for(int dc=-half;dc<=half;dc++){
            int idx = (dr+half)*kernel_size + (dc+half);
            weights[idx] = exp(-(dr*dr + dc*dc)/(2*sigma*sigma));
        }
    }
    // normalizing the weights such that they sum to 1
    double sum = 0;
    for(int i=0;i<weights.size();i++){
        sum += weights[i];
    }

    for(int i=0;i<weights.size();i++){
        weights[i] = weights[i]/sum;
    }


    for(int row = 0;row<height;row++){
        for(int col = 0; col<width;col++){
            double sum = 0;
            for(int dr=-half;dr<=half;dr++){
                for(int dc=-half;dc<=half;dc++){
                    int neighborRow = row + dr;
                    int neighborCol = col + dc;

                    neighborRow = clamp(neighborRow,0,height-1);
                    neighborCol = clamp(neighborCol,0,width-1);
                    int neighborVal = pixels[neighborRow*width + neighborCol];
                    int idx = (dr+half)*kernel_size + (dc+half);
                    sum += neighborVal*weights[idx];
                }
            }
            outPixels[row*width + col] = (uint8_t)sum;
        }
    }
    return result;
}

std::pair<py::array_t<uint8_t>, py::array_t<double>> sobel(py::array_t<uint8_t>input){
    py::buffer_info buf = input.request();

    uint8_t*pixels = reinterpret_cast<uint8_t*>(buf.ptr);
    int height = buf.shape[0];
    int width = buf.shape[1];

    py::array_t<double>result_dir({height,width});
    py::array_t<uint8_t>result_mag({height,width});
    uint8_t*outPixels = reinterpret_cast<uint8_t*>(result_mag.request().ptr);
    double*dir = reinterpret_cast<double*>(result_dir.request().ptr);

    int Gx[9] = {-1,0,1,-2,0,2,-1,0,1};
    int Gy[9] = {-1,-2,-1,0,0,0,1,2,1};

    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            double sumx = 0;
            double sumy = 0;
            for(int dr=-1;dr<=1;dr++){
                for(int dc=-1;dc<=1;dc++){
                    int neighborRow = row+dr;
                    int neighborCol = col+dc;
                    neighborRow = clamp(neighborRow,0,height-1);
                    neighborCol = clamp(neighborCol,0,width-1);

                    int neighborVal = pixels[neighborRow*width + neighborCol];
                    sumx += neighborVal * Gx[(dr+1)*3 + dc+1];
                    sumy += neighborVal * Gy[(dr+1)*3 + dc+1];
                }
            }
            int mag = sqrt(sumx*sumx + sumy*sumy);
            double tan_inverse = std::atan2(sumy,sumx);
            mag = clamp(mag,0,255);
            dir[row*width+col] = tan_inverse;
            outPixels[row*width+col] = (uint8_t)mag;
        }
    }
    return {result_mag,result_dir};
}

PYBIND11_MODULE(image_pipeline_cpp, m) {
    m.def("invert", &invert, "Inverts a grayscale image");
    m.def("to_grayscale", &to_grayscale, "Converts an RGB image to grayscale");
    m.def("threshold", &threshold, "Applies binary thresholding to a grayscale image");
    m.def("box_blur", &box_blur, "Applies a box blur to a grayscale image");
    m.def("gaussian_blur",&gaussian_blur, "Applies a Gaussian blur to a grayscale image");
    m.def("sobel", &sobel, "Computes Sobel gradient magnitude of a grayscale image");
}