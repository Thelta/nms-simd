// nms-simd.cpp : Defines the entry point for the application.
//

#include <vector>
#include <cstdint>
#include <immintrin.h>
#include <cmath>
#include <array>

struct Rect
{
	int x1;
	int x2;
	int y1;
	int y2;
};

__forceinline __m256 compareRectangles(size_t readIdx,
									   __m256i passX1_8,
									   __m256i passX2_8,
									   __m256i passY1_8,
									   __m256i passY2_8,
									   __m256i passArea_8,
									   __m256 threshold_8,
									   std::vector<int>& x1,
									   std::vector<int>& x2,
									   std::vector<int>& y1,
									   std::vector<int>& y2)
{
	const auto zero_8 = _mm256_set1_epi32(0);
	const auto one_8 = _mm256_set1_epi32(1);

	auto x1_8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(x1.data() + readIdx));
	auto x2_8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(x2.data() + readIdx));
	auto y1_8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(y1.data() + readIdx));
	auto y2_8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(y2.data() + readIdx));

	auto interX1_8 = _mm256_max_epi32(x1_8, passX1_8);
	auto interX2_8 = _mm256_min_epi32(x2_8, passX2_8);
	auto interY1_8 = _mm256_max_epi32(y1_8, passY1_8);
	auto interY2_8 = _mm256_min_epi32(y2_8, passY2_8);

	// (interX2 - interX1 + 1)
	auto interXDiff_8 = _mm256_sub_epi32(interX2_8, interX1_8);
	interXDiff_8 = _mm256_add_epi32(interXDiff_8, one_8);

	// (interY2 - interY1 + 1)
	auto interYDiff_8 = _mm256_sub_epi32(interY2_8, interY1_8);
	interYDiff_8 = _mm256_add_epi32(interYDiff_8, one_8);

	auto bboxWidth_8 = _mm256_max_epi32(zero_8, interXDiff_8);
	auto bboxHeight_8 = _mm256_max_epi32(zero_8, interYDiff_8);

	auto interArea_8 = _mm256_mullo_epi32(bboxWidth_8, bboxHeight_8);

	auto width_8 = _mm256_sub_epi32(x2_8, x1_8);
	auto height_8 = _mm256_sub_epi32(y2_8, y1_8);
	auto area_8 = _mm256_mullo_epi32(width_8, height_8);

	// area + passArea
	auto areaSum_8 = _mm256_add_epi32(area_8, passArea_8);
	// (area + passArea) - interArea
	areaSum_8 = _mm256_sub_epi32(areaSum_8, interArea_8);

	auto areaSum_8f = _mm256_cvtepi32_ps(areaSum_8);
	auto interArea_8f = _mm256_cvtepi32_ps(interArea_8);

	auto overlap_8 = _mm256_div_ps(interArea_8f, areaSum_8f);

	return _mm256_cmp_ps(threshold_8, overlap_8, 13);
}

std::vector<uint32_t> nmsSimd1(const std::vector<Rect> &rectangles, float threshold)
{
	size_t rectCount = rectangles.size();
	size_t simdRectCount = (rectCount + 7) / 8 * 8 + 7;

	std::vector<uint32_t> validnassMap(simdRectCount);
	for(size_t i = 0; i < rectCount; i++)
	{
		validnassMap[i] = 0xFFFFFFFF;
	}
	for(size_t i = rectCount; i < simdRectCount; i++)
	{
		validnassMap[i] = 0;
	}

	size_t passIdx = 0;
	size_t passCount = 1;

	std::vector<int> x1(simdRectCount), x2(simdRectCount), y1(simdRectCount), y2(simdRectCount);

	for(size_t i = 0; i < rectCount; i++)
	{
		x1[i] = rectangles[i].x1;
		x2[i] = rectangles[i].x2;
		y1[i] = rectangles[i].y1;
		y2[i] = rectangles[i].y2;
	}

	auto threshold_8 = _mm256_set1_ps(threshold);

	for(size_t pass = 0; pass < passCount; pass++)
	{
		bool isFound = false;

		auto passX1_8 = _mm256_set1_epi32(x1[passIdx]);
		auto passX2_8 = _mm256_set1_epi32(x2[passIdx]);
		auto passY1_8 = _mm256_set1_epi32(y1[passIdx]);
		auto passY2_8 = _mm256_set1_epi32(y2[passIdx]);

#if 0
		auto passWidth_8 = _mm256_sub_epi32(passX2_8, passX1_8);
		auto passHeight_8 = _mm256_sub_epi32(passY2_8, passY1_8);
		auto passArea_8 = _mm256_mullo_epi32(passWidth_8, passHeight_8);
#endif

#if 1
		auto passWidth = x2[passIdx] - x1[passIdx];
		auto passHeight = y2[passIdx] - y1[passIdx];
		auto passArea_8 = _mm256_set1_epi32(passWidth * passHeight);
#endif

		size_t passIdxTemp = passIdx;

		for(size_t readIdx = passIdxTemp + 1; readIdx < rectCount; readIdx += 8)
		{
			int control = false;
			auto validness_8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(validnassMap.data() + readIdx));
			control = _mm256_movemask_epi8(validness_8);
			const size_t remainingCount = 8;

			if(!control)
			{
				continue;
			}

			auto isValid_8 = compareRectangles(readIdx,
											   passX1_8,
											   passX2_8,
											   passY1_8,
											   passY2_8,
											   passArea_8,
											   threshold_8,
											   x1,
											   x2,
											   y1,
											   y2);

			auto isValid_8i = _mm256_cvtps_epi32(isValid_8);

			isValid_8i = _mm256_and_si256(isValid_8i, validness_8);
			auto isValid = _mm256_movemask_ps(_mm256_castsi256_ps(isValid_8i));
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(validnassMap.data() + readIdx), isValid_8i);

			if(!isFound && isValid)
			{
				for(size_t ele = 0; ele < remainingCount; ele++)
				{
					if(validnassMap[readIdx + ele])
					{
						isFound = true;
						passIdx = readIdx + ele;
						passCount++;
						break;
					}
				}
			}
		}
	}

	return validnassMap;
}