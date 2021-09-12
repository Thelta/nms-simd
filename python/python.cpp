#include "py-utils.h"
#include "nms-simd.h"
#include "nms-utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

PYBIND11_MODULE(py_nms_simd, m)
{
	m.doc() = "test";
	m.def("run", [](py::object all_rectangles, py::object scores, float score_threshold) {
		auto passRectIndices(NMS_SIMD_PY::runCreateRectangleIndices(scores, score_threshold));
		NMS_SIMD::Rectangles simdRects;
		NMS_SIMD::createRectangles(&simdRects, passRectIndices.size());
		NMS_SIMD_PY::runCopyBufferToRectangles(all_rectangles, simdRects, passRectIndices);
		NMS_SIMD::nmsSimd1(simdRects, score_threshold);
		py::list validIndices = NMS_SIMD_PY::getValidIndices(simdRects, passRectIndices);

		NMS_SIMD::destroyRectangles(&simdRects);

		return validIndices;
	});
}
