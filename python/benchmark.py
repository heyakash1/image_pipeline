import os
import sys
os.add_dll_directory(r"C:\mingw64\bin")
sys.path.append(r"C:\Users\akash\OneDrive\Desktop\image_pipeline\build")
import image_pipeline_cpp

import math
import numpy as np

def clamp(val,l,h):
    val = max(val,l)
    val = min(val,h)
    return val
def python_gaussian_blur(pixels, kernel_size, sigma):
    height = pixels.shape[0]
    width = pixels.shape[1]

    # // for integer division
    half = kernel_size//2

    n = kernel_size*kernel_size
    weights = np.zeros(n)

    for dr in range(-half,half+1):
        for dc in range(-half,half+1):
            idx = (dr+half)*kernel_size + (dc+half)
            weights[idx] = math.exp(-(dr**2 + dc**2)/(2*(sigma**2)))

    # normalizing weights
    weights_sum = 0
    for i in range(len(weights)):
        weights_sum += weights[i]

    weights = weights/weights_sum

    output = np.zeros((height,width),dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            updated_pixel = 0
            for dr in range(-half,half+1):
                for dc in range(-half,half+1):
                    neighbor_row = row + dr
                    neighbor_col = col + dc

                    neighbor_row = clamp(neighbor_row,0,height-1)
                    neighbor_col = clamp(neighbor_col,0,width-1)

                    neighbor_val = pixels[neighbor_row,neighbor_col]
                    idx = (dr+half)*kernel_size + (dc+half)
                    updated_pixel += neighbor_val*weights[idx]

            output[row,col] = updated_pixel
    return output


def numpy_gaussian_blur(pixels, kernel_size, sigma):
    height = pixels.shape[0]
    width = pixels.shape[1]
    
    # // for integer division
    half = kernel_size//2
    
    n = kernel_size*kernel_size
    weights = np.zeros(n)
    
    for dr in range(-half,half+1):
        for dc in range(-half,half+1):
            idx = (dr+half)*kernel_size + (dc+half)
            weights[idx] = math.exp(-(dr**2 + dc**2)/(2*(sigma**2)))
    
    # normalizing weights
    weights_sum = 0
    for i in range(len(weights)):
        weights_sum += weights[i]
    
    weights = weights/weights_sum

    padded = np.pad(pixels,half,mode='edge')
    result = np.zeros((height,width),dtype=float)

    for dr in range(-half,half+1):
        for dc in range(-half,half+1):
            shifted = padded[half+dr:half+dr+height,half+dc:half+dc+width]
            idx = (dr+half)*kernel_size + (dc+half)
            offset_multiply = shifted*weights[idx]
            result += offset_multiply

    return np.clip(result,0,255).astype(np.uint8)


import time
kernel_size = 9
sigma = 2.0

# loading image and the set-up
import matplotlib.pyplot as plt

# don't load bigger images as it takes too long
path = "C:/Users/akash/OneDrive/Desktop/image_pipeline/data/benchmark_test.jpg"
original_img = plt.imread(path)

if original_img.dtype != np.uint8:
    original_img = (original_img * 255).astype(np.uint8)

# Grayscale conversion now happens in C++
gray = image_pipeline_cpp.to_grayscale(original_img[:, :, :3])

start = time.perf_counter()
cpp_result = image_pipeline_cpp.gaussian_blur(gray,kernel_size,sigma)
cpp_time = time.perf_counter() - start

start = time.perf_counter()
python_result = python_gaussian_blur(gray,kernel_size,sigma)
python_time = time.perf_counter() - start

start = time.perf_counter()
numpy_result = numpy_gaussian_blur(gray,kernel_size,sigma)
numpy_time = time.perf_counter()-start

print(f"C++ time: {cpp_time:.4f}s")
print(f"Python time: {python_time:.4f}s")
print(f"NumPy-vectorized time: {numpy_time:.4f}s")
print(f"Naive Python speedup vs C++: {python_time / cpp_time:.1f}x slower")
print(f"NumPy speedup vs naive Python: {python_time / numpy_time:.1f}x faster")
print(f"C++ speedup vs NumPy:{numpy_time / cpp_time:.1f}x faster")

plt.figure(figsize=(7,5))
plt.bar(
    ["Naive Python","NumPy-vectorised","C++"],
    [python_time,numpy_time,cpp_time],
    color=["indianred","goldenrod","seagreen"]
)

plt.yscale('log')
plt.ylabel("Time (seconds,log scale)")
plt.title("Gaussian Blur: Naive Python vs NumPy vs C++ (Release build)")
plt.show()