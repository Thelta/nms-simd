#include <algorithm>
#include "nms-utils.h"

size_t NMS_SIMD::calculateSimdCount(size_t rectangleCount)
{
	return (rectangleCount + 7);
}

void NMS_SIMD::createRectangles(NMS_SIMD::Rectangles *rectangles, size_t rectangleCount)
{
	rectangles->count = rectangleCount;
	rectangles->simdCount = calculateSimdCount(rectangleCount);
	uint32_t* buffer = new uint32_t[5 * rectangles->simdCount];
	rectangles->x1 = buffer;
	rectangles->x2 = buffer + rectangles->simdCount;
	rectangles->y1 = buffer + rectangles->simdCount * 2;
	rectangles->y2 = buffer + rectangles->simdCount * 3;
	rectangles->validness = buffer + rectangles->simdCount * 4;

	for(size_t i = 0; i < rectangleCount; i++)
	{
		rectangles->validness[i] = 0xFFFFFFFF;
	}
	for(size_t i = rectangleCount; i < rectangles->simdCount; i++)
	{
		rectangles->validness[i] = 0;
	}
}

void NMS_SIMD::destroyRectangles(NMS_SIMD::Rectangles *rectangles)
{
	delete rectangles->x1;
	rectangles->x1 = nullptr;
	rectangles->x2 = nullptr;
	rectangles->y1 = nullptr;
	rectangles->y2 = nullptr;
	rectangles->validness = nullptr;
}

std::vector<size_t> NMS_SIMD::createRectangleIndices(const std::vector<float>& scores, float scoreThreshold)
{
	std::vector<size_t> passRectIndices;
	for(size_t i = 0; i < scores.size(); i++)
	{
		if(scores[i] >= scoreThreshold)
		{
			passRectIndices.push_back(i);
		}
	}

	std::sort(passRectIndices.begin(), passRectIndices.end(), [&](size_t a, size_t b) {
		return scores[a] > scores[b];
	});

	return passRectIndices;
}

std::vector<size_t> NMS_SIMD::getValidIndices(const NMS_SIMD::Rectangles& rectangles, const std::vector<size_t>& indices)
{
	std::vector<size_t> validIndices;
	for(size_t i = 0; i < rectangles.count; i++)
	{
		if(rectangles.validness[i] != 0)
		{
			validIndices.push_back(indices[i]);
		}
	}

	return validIndices;
}

std::vector<size_t> NMS_SIMD::createRectangleIndices(const float* scores, size_t scoreCount, float scoreThreshold)
{
	std::vector<size_t> passRectIndices;
	for(size_t i = 0; i < scoreCount; i++)
	{
		if(scores[i] >= scoreThreshold)
		{
			passRectIndices.push_back(i);
		}
	}

	std::sort(passRectIndices.begin(), passRectIndices.end(), [&](size_t a, size_t b) {
		return scores[a] > scores[b];
	});

	return passRectIndices;
}
