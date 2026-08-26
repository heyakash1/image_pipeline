import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import visualize


# --- Bridge setup ---------------------------------------------------
# Windows (3.8+) ignores PATH for locating DLL dependencies of C
# extension modules, so we must explicitly whitelist MinGW's runtime
# DLL folder before importing the compiled module.
os.add_dll_directory(r"C:\mingw64\bin")
sys.path.append(r"C:\Users\akash\OneDrive\Desktop\image_pipeline\build")
import image_pipeline_cpp

import argparse

parser = argparse.ArgumentParser(description="Canny edge detection.")
parser.add_argument("--image",required=True, help="Add an image")
parser.add_argument("--sigma",type=float, default=1.5, help="Set the value for sigma")
parser.add_argument("--kernel-size",type=int, default=5, help="Set the kernel_size for gaussian_blur")
parser.add_argument("--low",type=int, default=None, help="Low hysteresis threshold (optional: auto-computed if omitted)")
parser.add_argument("--high",type=int, default=None, help="High hysteresis threshold (optional: auto-computed if omitted)")
args = parser.parse_args()

if (args.low is None) != (args.high is None):
    parser.error("--low and --high must be provided together, or not at all")

# --- Load image -------------------------------------------------------
IMAGE_PATH = args.image
original_img = plt.imread(IMAGE_PATH)

if original_img.dtype != np.uint8:
    original_img = (original_img * 255).astype(np.uint8)

# Grayscale conversion now happens in C++
gray = image_pipeline_cpp.to_grayscale(np.ascontiguousarray(original_img[:, :, :3]))

blurred = image_pipeline_cpp.gaussian_blur(gray,args.kernel_size,args.sigma)
mag,direction = image_pipeline_cpp.sobel(blurred)

thinned = image_pipeline_cpp.non_max_suppression(mag,direction)

if args.low is not None and args.high is not None:
    low, high = args.low,args.high
else:
    low,high = image_pipeline_cpp.compute_adaptive_thresholds(mag)
final_edges = image_pipeline_cpp.hysteresis_threshold(thinned,low,high)

stages = [("grayscale",gray),("blurred",blurred),("sobel",mag),("NMS",thinned),("hysteresis",final_edges)]

visualize.show_pipeline_stages(stages)