#pragma once

#include <cstdint>
#include <cstddef>

namespace NMS_SIMD
{

struct Rectangles
{
	uint32_t* x1;
	uint32_t* x2;
	uint32_t* y1;
	uint32_t* y2;
	uint32_t* validness;
	size_t count; // aka size
	size_t simdCount; // aka capacity
};

}