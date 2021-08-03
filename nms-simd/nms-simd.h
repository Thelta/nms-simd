#pragma once
#include <vector>
#include <cstdint>

namespace NMS_SIMD
{
	struct Rect
	{
		int x1;
		int y1;
		int x2;
		int y2;
	};

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

	void nmsSimd1(const Rectangles& rects, float threshold);

	std::vector<size_t> runNmsSimd1(const std::vector<Rect>& rects, const std::vector<float>& scores, float scoreThreshold, float nmsThreshold);
}

