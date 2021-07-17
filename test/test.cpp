#include "nms-simd.h"

#include <gtest/gtest.h>

#include <opencv2/dnn/dnn.hpp>
#include <absl/random/random.h>

#include <vector>

constexpr size_t g_rectangleSize = 1000;
constexpr float g_scoreThreshold = 0.6;
constexpr float g_nmsThreshold = 0.4;

TEST(NMS, ValidnessTest)
{
	absl::BitGen bitgen;

	std::vector<int> rectanglesRaw(g_rectangleSize * 4);
	std::vector<float> scores(g_rectangleSize);

	for(size_t i = 0; i < g_rectangleSize; i++)
	{
		int x1 = absl::Uniform<int>(bitgen, 0, 1200);
		int y1 = absl::Uniform<int>(bitgen, 0, 1200);
		int x2 = x1 + absl::Uniform<int>(bitgen, 20, 60);
		int y2 = y1 + absl::Uniform<int>(bitgen, 20, 60);

		rectanglesRaw[i * 4] = x1;
		rectanglesRaw[i * 4 + 1] = y1;
		rectanglesRaw[i * 4 + 2] = x2;
		rectanglesRaw[i * 4 + 3] = y2;

		scores[i] = absl::Uniform<float>(bitgen, 0, 1);
	}
#if 0
	std::vector<cv::Rect> cvRectangles(g_rectangleSize);
	for(size_t i = 0; i < g_rectangleSize; i++)
	{
		cvRectangles[i] = cv::Rect(
			rectanglesRaw[i * 4],
			rectanglesRaw[i * 4 + 1],
			rectanglesRaw[i * 4 + 2],
			rectanglesRaw[i * 4 + 3]
		);
	}

	cv::dnn::MatShape cvIndices;
	for(size_t i = 0; i < 100000; i++)
	{
		cv::dnn::NMSBoxes(cvRectangles, scores, 0, g_nmsThreshold, cvIndices);
		cvIndices.clear();
	}
#endif

#if 2
	std::vector<Rect> simdRectangles(g_rectangleSize);
	std::copy(rectanglesRaw.data(), rectanglesRaw.data() + rectanglesRaw.size(), (int*)simdRectangles.data());

	size_t a = 1;
	for(size_t i = 0; i < 1; i++)
	{
		nmsSimd1(simdRectangles, g_nmsThreshold);
	}
	std::cout << a << std::endl;
#endif
}