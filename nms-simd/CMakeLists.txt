# CMakeList.txt : CMake project for nms-simd, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.8)

# Add source to this project's executable.
add_library(nms-simd 
	nms-simd.h
	rectangles.h
	nms-simd.cpp
	nms-utils.h
	nms-utils.cpp
)
