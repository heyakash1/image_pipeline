import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
# --- Bridge setup ---------------------------------------------------
# Windows (3.8+) ignores PATH for locating DLL dependencies of C
# extension modules, so we must explicitly whitelist MinGW's runtime
# DLL folder before importing the compiled module.
os.add_dll_directory(r"C:\mingw64\bin")
sys.path.append(r"C:\Users\akash\OneDrive\Desktop\image_pipeline\build")
import image_pipeline_cpp

# --- Load image -------------------------------------------------------
IMAGE_PATH = "C:/Users/akash/OneDrive/Desktop/image_pipeline/text_image.jpg"
original_img = plt.imread(IMAGE_PATH)

if original_img.dtype != np.uint8:
    original_img = (original_img * 255).astype(np.uint8)

# --- Grayscale conversion (Phase 1 stopgap; C++ will own this later) --
# gray = original_img[:, :, :3].mean(axis=2).astype(np.uint8)


# Grayscale conversion now happens in C++
gray = image_pipeline_cpp.to_grayscale(original_img[:, :, :3])
inverted_img = image_pipeline_cpp.invert(gray)
thresholded_img = image_pipeline_cpp.threshold(gray,128)

blurred = image_pipeline_cpp.gaussian_blur(gray,5,1.5)
mag,direction = image_pipeline_cpp.sobel(blurred)

thinned = image_pipeline_cpp.non_max_suppression(mag,direction)
final_edges = image_pipeline_cpp.hysteresis_threshold(thinned,50,100)
cv_edges = cv2.Canny(blurred,50,100)
# print("mag non-zero pixels:",np.count_nonzero(mag))
# print("thinned non-zero pixels:",np.count_nonzero(thinned))
# print("thinned max value:",thinned.max())
# print("mag dtype:",mag.dtype,"shape:",mag.shape)
# print("direction dtype:",direction.dtype,"shape:",direction.shape)
# --- Visualize ----------------------------------------------------------
# fig, axes = plt.subplots(1, 5, figsize=(14, 5))
# for ax, img, title in [
#     (axes[0], gray, "Grayscale (C++)"),
#     (axes[1], blurred, "Gaussian (25*25) (C++)"),
#     (axes[2], mag, "sobel gradient (C++)"),
#     (axes[3], thinned, "non-maximum suppression (C++)"),
#     (axes[4], final_edges, "hysteresis thresholding (C++)"),
# ]:
#     ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
#     ax.set_title(title)
#     ax.axis("off")

# print("thinned non-zero:",np.count_nonzero(thinned))
# print("final_edges non-zero:",np.count_nonzero(final_edges))
# print("final_edges unique values:",np.unique(final_edges))

# plt.figure(figsize=(14,9))
# plt.imshow(final_edges,cmap="gray")
# plt.axis('off')
# plt.title("Final Canny-style edges")
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(final_edges, cmap="gray")
axes[0].set_title("Your Canny")
axes[0].axis("off")

axes[1].imshow(cv_edges, cmap="gray")
axes[1].set_title("OpenCV Canny")
axes[1].axis("off")

plt.show()
# plt.figure(figsize=(10,6))
# plt.imshow(thinned,cmap="gray")
# plt.axis('off')