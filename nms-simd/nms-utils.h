#pragma once

#include "rectangles.h"

#include <vector>

namespace NMS_SIMD
{

size_t calculateSimdCount(size_t rectangleCount);
void createRectangles(Rectangles *rectangles, size_t rectangleCount);
void destroyRectangles(Rectangles *rectangles);

std::vector<size_t> createRectangleIndices(const std::vector<float>& scores, float scoreThreshold);
std::vector<size_t> createRectangleIndices(const float* scores, size_t scoreCount, float scoreThreshold);
std::vector<size_t> getValidIndices(const NMS_SIMD::Rectangles& rectangles, const std::vector<size_t>& indices);

}