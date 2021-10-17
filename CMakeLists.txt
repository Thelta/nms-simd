# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required(VERSION 3.8)

option(NMS_SIMD_BUILD_TEST OFF)
option(NMS_SIMD_BUILD_PYTHON OFF)
option(NMS_SIMD_BUILD_BENCHMARK OFF)

project("nms-simd")

# Include sub-projects.
add_subdirectory("nms-simd")
if(${NMS_SIMD_BUILD_TEST})
	add_subdirectory("test")
endif ()
if(${NMS_SIMD_BUILD_BENCHMARK})
	add_subdirectory("benchmark")
endif()

if(${NMS_SIMD_BUILD_PYTHON})
	add_subdirectory("python")
endif()