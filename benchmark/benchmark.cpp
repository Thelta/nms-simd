#include <benchmark/benchmark.h>
#include <pcg_random.hpp>
#include <random>
#include <opencv2/dnn/dnn.hpp>

#include "nms-simd.h"
#include "nms-utils.h"

constexpr size_t g_rectangleSize = 1024;
constexpr float g_scoreThreshold = 0.6;
constexpr float g_nmsThreshold = 0.4;

pcg64 g_rng(42u, 54u);

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

std::vector<int> createRawRects(pcg64 &rng)
{
	std::vector<int> rectanglesRaw(g_rectangleSize * 4);
	std::uniform_int_distribution<int> pointDist(0, 1200);
	std::uniform_int_distribution<int> sideDist(20, 60);
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

	return rectanglesRaw;
}

std::vector<float> createScores(pcg64& rng)
{
	std::vector<float> scores(g_rectangleSize);

	std::uniform_real_distribution<float> scoreDist(0, 1);
	for(size_t i = 0; i < g_rectangleSize; i++)
	{
		scores[i] = scoreDist(rng);
	}

	return scores;
}


const std::vector<int> g_rawRectangles(createRawRects(g_rng));
const std::vector<float> g_scores(createScores(g_rng));

static void BM_nms_simd(benchmark::State& state)
{
	using namespace NMS_SIMD;
	std::vector<Rect> simdRectangles(g_rectangleSize);
	std::copy(g_rawRectangles.data(), g_rawRectangles.data() + g_rawRectangles.size(), (int*)simdRectangles.data());

	for(auto _ : state)
	{
		runNmsSimd1(simdRectangles, g_scores, 0, g_nmsThreshold);
	}
}

static void BM_cv_nms(benchmark::State& state)
{
	std::vector<cv::Rect> cvRectangles(g_rectangleSize);
	for(size_t i = 0; i < g_rectangleSize; i++)
	{
		cvRectangles[i] = cv::Rect(
			g_rawRectangles[i * 4],
			g_rawRectangles[i * 4 + 1],
			g_rawRectangles[i * 4 + 2] - g_rawRectangles[i * 4],
			g_rawRectangles[i * 4 + 3] - g_rawRectangles[i * 4 + 1]
		);
	}

	for(auto _ : state)
	{
		cv::dnn::MatShape cvIndices;
		cv::dnn::NMSBoxes(cvRectangles, g_scores, 0, g_nmsThreshold, cvIndices);
	}
}

BENCHMARK(BM_nms_simd);
BENCHMARK(BM_cv_nms);

BENCHMARK_MAIN();