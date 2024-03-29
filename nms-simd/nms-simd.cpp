#include "nms-simd.h"

#include <immintrin.h>

#if defined(__GNUC__)
#define NMS_INLINE __attribute__((always_inline)) inline
#elif defined(__clang__)
#define NMS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSVC_LANG)
#define NMS_INLINE __forceinline
#else
#define NMS_INLINE
#warning "Unknown compiler, can't force inlining."
#endif

NMS_INLINE static __m256 compareRectangles(size_t readIdx,
										   __m256i passX1_8,
										   __m256i passX2_8,
										   __m256i passY1_8,
										   __m256i passY2_8,
										   __m256i passArea_8,
										   __m256i x1_8,
										   __m256i x2_8,
										   __m256i y1_8,
										   __m256i y2_8,
										   __m256 threshold_8)
{
	const auto zero_8 = _mm256_set1_epi32(0);
	const auto one_8 = _mm256_set1_epi32(1);

	auto interX1_8 = _mm256_max_epi32(x1_8, passX1_8);
	auto interX2_8 = _mm256_min_epi32(x2_8, passX2_8);
	auto interY1_8 = _mm256_max_epi32(y1_8, passY1_8);
	auto interY2_8 = _mm256_min_epi32(y2_8, passY2_8);

	// (interX2 - interX1)
	auto interXDiff_8 = _mm256_sub_epi32(interX2_8, interX1_8);

	// (interY2 - interY1)
	auto interYDiff_8 = _mm256_sub_epi32(interY2_8, interY1_8);

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

void NMS_SIMD::nmsSimd1(const Rectangles& rects, float threshold)
{
	size_t passIdx = 0;
	size_t passCount = 1;

	auto threshold_8 = _mm256_set1_ps(threshold);

	for(size_t pass = 0; pass < passCount; pass++)
	{
		bool isFound = false;

		auto passX1_8 = _mm256_set1_epi32(rects.x1[passIdx]);
		auto passX2_8 = _mm256_set1_epi32(rects.x2[passIdx]);
		auto passY1_8 = _mm256_set1_epi32(rects.y1[passIdx]);
		auto passY2_8 = _mm256_set1_epi32(rects.y2[passIdx]);

		auto passWidth = rects.x2[passIdx] - rects.x1[passIdx];
		auto passHeight = rects.y2[passIdx] - rects.y1[passIdx];
		auto passArea_8 = _mm256_set1_epi32(passWidth * passHeight);

		size_t passIdxTemp = passIdx;

		for(size_t readIdx = passIdxTemp + 1; readIdx < rects.count; readIdx += 8)
		{
			int control = false;
			auto validness_8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(rects.validness + readIdx));
			control = _mm256_movemask_epi8(validness_8);
			const size_t remainingCount = 8;

			if(!control)
			{
				continue;
			}

			auto x1_8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(rects.x1 + readIdx));
			auto x2_8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(rects.x2 + readIdx));
			auto y1_8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(rects.y1 + readIdx));
			auto y2_8 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(rects.y2 + readIdx));

			auto isValid_8 = compareRectangles(readIdx,
											   passX1_8,
											   passX2_8,
											   passY1_8,
											   passY2_8,
											   passArea_8,
											   x1_8,
											   x2_8,
											   y1_8,
											   y2_8,
											   threshold_8);

			auto isValid_8i = _mm256_cvtps_epi32(isValid_8);

			isValid_8i = _mm256_and_si256(isValid_8i, validness_8);
			_mm256_storeu_si256(reinterpret_cast<__m256i*>(rects.validness + readIdx), isValid_8i);

			int isValid;
			if (!isFound && (isValid = _mm256_movemask_ps(_mm256_castsi256_ps(isValid_8i))))
			{
				isFound = true;
				passIdx = _tzcnt_u32(isValid) + readIdx;
				passCount++;
			}
		}
	}
}
