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
if(${MSVC})
	target_compile_options(nms-simd PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
	target_compile_options(nms-simd PRIVATE $<$<CONFIG:Release>:/O2>)
else()
	target_compile_options(nms-simd PRIVATE "-mavx2")
	target_compile_options(nms-simd PRIVATE $<$<CONFIG:Release>:-O2>)
endif ()

