# Image Processing Pipeline (C++ / Python / pybind11)

A from-scratch Canny edge detector and image processing pipeline, with performance-critical
algorithms written in C++ and exposed to Python via pybind11. Built as a first solo project
to learn C++/Python interop, convolution-based image algorithms, and honest performance
benchmarking.

## What it does

- Grayscale conversion, image inversion, binary thresholding
- Box blur and Gaussian blur (clamped borders)
- Sobel gradient computation (magnitude + direction, returned together via `std::pair`)
- Non-maximum suppression (4-directional gradient bucketing)
- Hysteresis thresholding (BFS flood-fill to link weak edges to strong ones)
- Histogram matching (CDF-based distribution remapping between two images)

Together, the first five form a complete, hand-built Canny edge detector — validated
against `cv2.Canny()`.

## Example output

<img width="2228" height="734" alt="image" src="https://github.com/user-attachments/assets/731f3e5c-0567-44c3-9dcb-11388f17e5a5" />


## Performance

Gaussian blur, same algorithm, three implementations:

| Implementation       | Time    | Relative to C++  |
|----------------------|---------|------------------|
| Naive Python loop    | 27.9s   | ~590x slower     |
| NumPy-vectorized     | 0.21s   | ~4.4x slower     |
| C++ (via pybind11)   | 0.047s  | baseline         |

<img width="1352" height="1048" alt="image" src="https://github.com/user-attachments/assets/3e4c16c9-af45-43bd-8ccc-ac64ed43bd2e" />


Note: the C++ build initially ran *slower* than the vectorized NumPy version until I
discovered the build wasn't using `-O2`/`-O3` optimizations. Rebuilding with
`-DCMAKE_BUILD_TYPE=Release` gave roughly an 11x speedup on its own — a good reminder
that "written in C++" alone doesn't guarantee performance; build configuration matters
just as much as language choice.

## Project structure

<img width="1038" height="356" alt="image" src="https://github.com/user-attachments/assets/e4bab0f1-4343-40d5-88e8-d4bab42a09bd" />

## Building it

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

Requires: a C++ compiler (MinGW/g++ or MSVC), CMake, Python 3.8+, `pybind11`, `numpy`,
`matplotlib`.

## Running it

```bash
cd python
python pipeline.py
```

## What I learned

- Bridging C++ and Python with pybind11: `py::array_t`, buffer protocols, and returning
  multiple values via `std::pair`
- Convolution fundamentals: kernels, border handling (clamping vs. zero-pad vs. reflect),
  and why intermediate computation types matter (avoiding `uint8` overflow)
- BFS/flood-fill for connectivity-based logic (hysteresis thresholding), as opposed to the
  fixed-neighborhood convolutions used everywhere else in the pipeline
- Real Windows-specific debugging: Smart App Control blocking unsigned DLLs, and the
  Python 3.8+ change where `PATH` is no longer used to locate extension-module DLL
  dependencies (`os.add_dll_directory` instead)
- Benchmarking honestly: comparing against a *fair* baseline (vectorized NumPy) rather
  than only a strawman (naive loops), and verifying build configuration before trusting
  a performance number
