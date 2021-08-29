#include "nms-simd.h"
#include "nms-utils.h"

#include <gtest/gtest.h>

#include <opencv2/dnn/dnn.hpp>
#include <random>
#include <vector>

#include <pcg_random.hpp>

constexpr size_t g_rectangleSize = 1000;
constexpr float g_scoreThreshold = 0.6;
constexpr float g_nmsThreshold = 0.4;

struct Rect
{
	int x1;
	int y1;
	int x2;
	int y2;
};

std::vector<size_t> runNmsSimd1(const std::vector<Rect>& rects, const std::vector<float>& scores, float scoreThreshold, float nmsThreshold)
{
	std::vector<size_t> passRectIndices = NMS_SIMD::createRectangleIndices(scores, scoreThreshold);

	NMS_SIMD::Rectangles simdRects;
	NMS_SIMD::createRectangles(&simdRects, passRectIndices.size());

	for(size_t i = 0; i < passRectIndices.size(); i++)
	{
		size_t rectIdx = passRectIndices[i];
		simdRects.x1[i] = rects[rectIdx].x1;
		simdRects.x2[i] = rects[rectIdx].x2;
		simdRects.y1[i] = rects[rectIdx].y1;
		simdRects.y2[i] = rects[rectIdx].y2;
	}

	nmsSimd1(simdRects, nmsThreshold);

	std::vector<size_t> indices = NMS_SIMD::getValidIndices(simdRects, passRectIndices);

	NMS_SIMD::destroyRectangles(&simdRects);

	return indices;
}

TEST(NMS, SyntheticData)
{
	using namespace NMS_SIMD;
	std::vector<int> rectanglesRaw(g_rectangleSize * 4);
	std::vector<float> scores(g_rectangleSize);

	pcg64 rng(42u, 54u);
	std::uniform_int_distribution<int> pointDist(0, 1200);
	std::uniform_int_distribution<int> sideDist(20, 60);
	std::uniform_real_distribution<float> scoreDist(0, 1);

	for(size_t i = 0; i < g_rectangleSize; i++)
	{
		int x1 = pointDist(rng);
		int y1 = pointDist(rng);
		int x2 = x1 + sideDist(rng);
		int y2 = y1 + sideDist(rng);

		rectanglesRaw[i * 4] = x1;
		rectanglesRaw[i * 4 + 1] = y1;
		rectanglesRaw[i * 4 + 2] = x2;
		rectanglesRaw[i * 4 + 3] = y2;
	}

	for(size_t i = 0; i < g_rectangleSize; i++)
	{
		scores[i] = scoreDist(rng);
	}

	std::vector<cv::Rect> cvRectangles(g_rectangleSize);
	for(size_t i = 0; i < g_rectangleSize; i++)
	{
		cvRectangles[i] = cv::Rect(
			rectanglesRaw[i * 4],
			rectanglesRaw[i * 4 + 1],
			rectanglesRaw[i * 4 + 2] - rectanglesRaw[i * 4],
			rectanglesRaw[i * 4 + 3] - rectanglesRaw[i * 4 + 1]
		);
	}

	cv::dnn::MatShape cvIndices;
	cv::dnn::NMSBoxes(cvRectangles, scores, g_scoreThreshold, g_nmsThreshold, cvIndices);

	std::vector<Rect> simdRectangles(g_rectangleSize);
	std::copy(rectanglesRaw.data(), rectanglesRaw.data() + rectanglesRaw.size(), (int*)simdRectangles.data());
	auto result = runNmsSimd1(simdRectangles, scores, g_scoreThreshold, g_nmsThreshold);

	ASSERT_EQ(result.size(), cvIndices.size());
	for(size_t i = 0; i < result.size(); i++)
	{
		EXPECT_EQ(result[i], cvIndices[i]) << "idx " << i;
	}
}