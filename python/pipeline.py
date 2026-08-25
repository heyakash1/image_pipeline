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

# --- Load image -------------------------------------------------------
IMAGE_PATH = "C:/Users/akash/Downloads/pexels-edgar-arroyo-418919769-15497599.jpg"
original_img = plt.imread(IMAGE_PATH)

if original_img.dtype != np.uint8:
    original_img = (original_img * 255).astype(np.uint8)

# Grayscale conversion now happens in C++
gray = image_pipeline_cpp.to_grayscale(np.ascontiguousarray(original_img[:, :, :3]))

# histogram matching
# reference_gray = np.clip(gray.astype(int) + 80, 0, 255).astype(np.uint8)
# matched = image_pipeline_cpp.histogram_matching(gray,reference_gray)

gaussian_blur = image_pipeline_cpp.gaussian_blur(gray,5,1.5)
mag,direction = image_pipeline_cpp.sobel(gaussian_blur)

thinned = image_pipeline_cpp.non_max_suppression(mag,direction)
final_edges = image_pipeline_cpp.hysteresis_threshold(thinned,15,30)

stages = [("grayscale",gray),("blurred",gaussian_blur),("sobel",mag),("NMS",thinned),("hysteresis",final_edges)]
# visualize.show_pipeline_stages(stages)

# print("Original mean brightness:",gray.mean())
# print("Reference mean brightness:",reference_gray.mean())
# print("Matched mean brightness:",matched.mean())

# print("Matched histogram (sample):",np.histogram(matched,bins=8)[0])
# print("Reference histogram (sample):",np.histogram(reference_gray,bins=8)[0])

# print(original_img.shape,original_img.dtype)
# sliced = original_img[:,:,:3]
# print(sliced.flags['C_CONTIGUOUS'])
print("Sobel mag stats — min:", mag.min(), "max:", mag.max(), "mean:", mag.mean())
print("Thinned (NMS) stats — min:", thinned.min(), "max:", thinned.max(), "mean:", thinned.mean())
print("Non-zero in thinned:", np.count_nonzero(thinned))
print("Non-zero in final_edges:", np.count_nonzero(final_edges))
visualize.show_pipeline_stages([
    ("Grayscale",gray),
    ("Blurred",gaussian_blur),
    ("Sobel",mag),
    ("NMS",thinned),
    ("Hysteresis",final_edges),
])