cmake_minimum_required(VERSION 3.15)
project(image_pipeline)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

message(STATUS "pybind11 version: ${pybind11_VERSION}")

pybind11_add_module(image_pipeline_cpp cpp/src/bindings.cpp)