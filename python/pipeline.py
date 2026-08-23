import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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

blurred_3 = image_pipeline_cpp.box_blur(gray,3)
blurred_9 = image_pipeline_cpp.box_blur(gray,9)
blurred_25 = image_pipeline_cpp.box_blur(gray,25)

# print("gray vs blurred_2 identical?",np.array_equal(gray,blurred_3))
# print("max difference: ",np.max(np.abs(gray.astype(int)-blurred_3.astype(int))))
print("max difference (25*25): ",np.max(np.abs(gray.astype(int)-blurred_25.astype(int))))
# --- Visualize ----------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
for ax, img, title in [
    (axes[0], gray, "Grayscale (C++)"),
    (axes[1], blurred_3, "Blurred (3*3) (C++)"),
    (axes[2], blurred_9, "Blurred (9*9) (C++)"),
    (axes[3], blurred_25, "Blurred (25*25) (C++)"),
]:
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax.set_title(title)
    ax.axis("off")

plt.show()