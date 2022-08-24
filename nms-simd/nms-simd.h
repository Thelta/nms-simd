#pragma once
#include "rectangles.h"

#include <vector>
#include <cstdint>

namespace NMS_SIMD
{
	std::vector<int32_t> nmsSimd1(const Rectangles& rects, float threshold);
}

