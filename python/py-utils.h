#pragma once

#include "rectangles.h"

#include <vector>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>

namespace NMS_SIMD_PY
{

namespace py = pybind11;

std::vector<size_t> createRectangleIndices(const pybind11::list& scores, float scoreThreshold);
std::vector<size_t> createRectangleIndices(const pybind11::buffer& scores, float scoreThreshold);
pybind11::list getValidIndices(const NMS_SIMD::Rectangles& rectangles, const std::vector<size_t>& indices);
template<class T>
void copyBufferToRectangles(T* buffer, NMS_SIMD::Rectangles rectangles, const std::vector<size_t>& indices);
void copyBufferToRectangles(pybind11::list buffer, NMS_SIMD::Rectangles rectangles, const std::vector<size_t>& indices);
std::vector<size_t> runCreateRectangleIndices(pybind11::object scores, float scoreThreshold);
void runCopyBufferToRectangles(pybind11::object buffer, NMS_SIMD::Rectangles rectangles, const std::vector<size_t>& indices);
}

