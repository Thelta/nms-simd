#include "py-utils.h"

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace TestVals
{
namespace RectangleIndices
{

constexpr std::array<float, 20> scores {
	0.057231065777490375f,
	0.6224186353986474f,
	0.02605620292429789f,
	0.709925942057326f,
	0.07875775269324115f,
	0.28867305395811993f,
	0.824067369721239f,
	0.22436784297210632f,
	0.740745552865399f,
	0.20988869363098028f,
	0.8728412139500749f,
	0.5606433691121198f,
	0.6622035873721849f,
	0.5941905137885252f,
	0.6053420015768238f,
	0.244501392842562f,
	0.39459139821342526f,
	0.6352763922515868f,
	0.9927896152565082f,
	0.6409131246784319f
};

constexpr  std::array<size_t, 13> result {
		18, 10,  6,  8,  3, 12, 19, 17,  1, 14, 13, 11, 16
};

constexpr float threshold = 0.3f;
}
// namespace RectangleIndices
}
// namespace TestVals

TEST(PY_UTILS, RectangleIndicesList)
{
	using namespace TestVals::RectangleIndices;
	py::scoped_interpreter guard{};
	auto pylistScores = py::list(py::cast(scores));

	auto testResults = NMS_SIMD_PY::runCreateRectangleIndices(pylistScores, threshold);

	ASSERT_EQ(testResults.size(), result.size());
	for(size_t i = 0; i < result.size(); i++)
	{
		EXPECT_EQ(result[i], testResults[i]);
	}
}

TEST(PY_UTILS, RectangleIndicesBuffer)
{
	using namespace TestVals::RectangleIndices;
	py::scoped_interpreter guard{};
	auto pybufferScores = py::array_t<float>(py::cast(scores));

	auto testResults = NMS_SIMD_PY::runCreateRectangleIndices(pybufferScores, threshold);

	ASSERT_EQ(testResults.size(), result.size());
	for(size_t i = 0; i < result.size(); i++)
	{
		EXPECT_EQ(result[i], testResults[i]);
	}
}

TEST(PY_UTILS, CopyBufferList)
{
	pcg64 rng(42u, 54u);
	std::uniform_int_distribution<int> pointDist(0, 1200);
	std::uniform_int_distribution<int> sideDist(20, 60);
	std::uniform_real_distribution<float> scoreDist(0, 1);

}