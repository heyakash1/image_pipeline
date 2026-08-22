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
gray = original_img[:, :, :3].mean(axis=2).astype(np.uint8)

# --- Run through C++ ---------------------------------------------------
inverted_img = image_pipeline_cpp.invert(gray)

# --- Visualize ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, img, title, cmap in [
    (axes[0], original_img, "Original Image", None),
    (axes[1], inverted_img, "Inverted by C++", "gray"),
]:
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")

plt.show()