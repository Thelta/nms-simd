project(py_nms_simd)

find_package(pybind11 CONFIG REQUIRED)

add_library(
	py_utils
	py-utils.h
	py-utils.cpp
)

set_target_properties(py_utils PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(
	py_utils
	pybind11::module
)
target_include_directories(
	py_utils
	PRIVATE "../nms-simd"
)

pybind11_add_module(
	py_nms_simd
	python.cpp
)

target_include_directories(
	py_nms_simd
	PRIVATE "../nms-simd"
)

target_link_libraries(py_nms_simd PRIVATE
	pybind11::lto
	pybind11::embed
	pybind11::module
	nms-simd
	py_utils
)