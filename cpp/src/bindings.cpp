#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>
#include <queue>
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

py::array_t<uint8_t> non_max_suppression(py::array_t<uint8_t> mag_input, py::array_t<double>dir_input){
    py::buffer_info mag_buff = mag_input.request();
    py::buffer_info dir_buff = dir_input.request();

    uint8_t*mag = reinterpret_cast<uint8_t*>(mag_buff.ptr);
    double*dir = reinterpret_cast<double*>(dir_buff.ptr);

    int height = mag_buff.shape[0];
    int width = mag_buff.shape[1];

    py::array_t<uint8_t>result({height,width});

    uint8_t*outPixels = reinterpret_cast<uint8_t*>(result.request().ptr);

    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            int idx = row*width + col;

            double degrees = dir[idx] * 180.0/M_PI;
            if(degrees < 0)     degrees += 180;

            int neighborRow1,neighborRow2;
            int neighborCol1,neighborCol2;
            if((degrees >=0 && degrees <=22.5) || degrees>=157.5 && degrees<=180){
                neighborRow1 = row;
                neighborCol1 = col-1;
                neighborRow2 = row;
                neighborCol2 = col+1;
            }
            else if(degrees >=22.5 && degrees <=67.5){
                neighborRow1 = row-1;
                neighborCol1 = col+1;
                neighborRow2 = row+1;
                neighborCol2 = col-1;
            }
            else if(degrees >=67.5 && degrees <=112.5){
                neighborRow1 = row-1;
                neighborCol1 = col;
                neighborRow2 = row+1;
                neighborCol2 = col;
            }
            else{
                neighborRow1 = row-1;
                neighborCol1 = col-1;
                neighborRow2 = row+1;
                neighborCol2 = col+1;
            }
            neighborRow1 = clamp(neighborRow1,0,height-1);
            neighborCol1 = clamp(neighborCol1,0,width-1);
            neighborRow2 = clamp(neighborRow2,0,height-1);
            neighborCol2 = clamp(neighborCol2,0,width-1);
            if(mag[idx]>=mag[neighborRow1*width + neighborCol1] && mag[idx]>=mag[neighborRow2*width + neighborCol2]){
                outPixels[idx] = mag[idx];
            }
            else    outPixels[idx] = 0;
        }
    }
    return result;
}

py::array_t<uint8_t>hysteresis_threshold(py::array_t<uint8_t>input, uint8_t low, uint8_t high){
    py::buffer_info buf = input.request();

    uint8_t*pixels = reinterpret_cast<uint8_t*>(buf.ptr);
    int height = buf.shape[0];
    int width = buf.shape[1];

    py::array_t<uint8_t>result({height,width});
    uint8_t*outPixels = reinterpret_cast<uint8_t*>(result.request().ptr);

    std::queue<std::pair<int,int>>q;

    for(int row=0;row<height;row++){
        for(int col=0;col<width;col++){
            int idx = row*width + col;

            // strong
            if(pixels[idx] >= high){
                outPixels[idx] = 255;
                q.push({row,col});
            }
            // weak
            else if(pixels[idx]>=low){
                outPixels[idx] = 128;
            }
            // none
            else outPixels[idx] = 0;
        }
    }

    // flood-fill
    while(!q.empty()){
        auto front = q.front();
        q.pop();

        int row = front.first;
        int col = front.second;

        for(int dr=-1;dr<=1;dr++){
            for(int dc=-1;dc<=1;dc++){
                int nRow = row + dr;
                int nCol = col + dc;

                if(nRow >=0 && nRow<height && nCol>=0 && nCol<width){
                    int nIdx = nRow*width + nCol;
    
                    if(outPixels[nIdx]==128){
                        outPixels[nIdx] = 255;
                        q.push({nRow,nCol});
                    }
                }
            }
        }
    }

    int iterations = height*width;
    for(int i=0;i<iterations;i++){
        if(outPixels[i]==128)   outPixels[i] = 0;
    }
    return result;
}

py::array_t<uint8_t> histogram_matching(py::array_t<uint8_t>input, py::array_t<uint8_t>reference){
    py::buffer_info buf = input.request();
    py::buffer_info buf_ref = reference.request();
    int height = buf.shape[0];
    int width = buf.shape[1];

    int height_ref = buf_ref.shape[0];
    int width_ref = buf_ref.shape[1];
    uint8_t*pixels = reinterpret_cast<uint8_t*>(buf.ptr);
    uint8_t*reference_pixels = reinterpret_cast<uint8_t*>(buf_ref.ptr);

    py::array_t<uint8_t>result ({height,width});
    uint8_t*out_pixels = reinterpret_cast<uint8_t*>(result.request().ptr);

    std::vector<int>histogram(256,0);
    std::vector<int>histogram_ref(256,0);
    int iterations = height*width;
    int iterations2 = height_ref*width_ref;
    for(int i=0;i<iterations;i++){
        histogram[pixels[i]] ++;
    }
    
    for(int i=0;i<iterations2;i++){
        histogram_ref[reference_pixels[i]]++;
    }

    std::vector<int>cdf(256,0);
    cdf[0] = histogram[0];
    std::vector<int>cdf_ref(256,0);
    cdf_ref[0] = histogram_ref[0];
    for(int i=1;i<256;i++){
        cdf[i] = cdf[i-1] + histogram[i];
        cdf_ref[i] = cdf_ref[i-1] + histogram_ref[i];
    }

    // mapping array
    std::vector<uint8_t>lookup(256);
    for(int v=0;v<256;v++){
        int r_val = -1;
        int mini = 1e9;
        for(int r=0;r<256;r++){
            if(mini > abs(cdf_ref[r]-cdf[v])){
                mini = abs(cdf_ref[r]-cdf[v]);
                r_val = r;
            }
        }
        lookup[v] = r_val;
    }

    for(int i=0;i<iterations;i++){
        out_pixels[i] = lookup[pixels[i]];
    }
    return result;
}
PYBIND11_MODULE(image_pipeline_cpp, m) {
    m.def("invert", &invert, "Inverts a grayscale image");
    m.def("to_grayscale", &to_grayscale, "Converts an RGB image to grayscale");
    m.def("threshold", &threshold, "Applies binary thresholding to a grayscale image");
    m.def("box_blur", &box_blur, "Applies a box blur to a grayscale image");
    m.def("gaussian_blur",&gaussian_blur, "Applies a Gaussian blur to a grayscale image");
    m.def("sobel", &sobel, "Computes Sobel gradient magnitude of a grayscale image");
    m.def("non_max_suppression", &non_max_suppression, "Applies non-maximum suppression to thin sobel edges");
    m.def("hysteresis_threshold",&hysteresis_threshold, "Applies hysteresis thresholding to link edge segments");
    m.def("histogram_matching",&histogram_matching, "Matches the histogram of an image to a reference image");
}