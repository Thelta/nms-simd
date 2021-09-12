#include "py-utils.h"

#include <algorithm>
#include <fmt/format.h>
#include <pybind11/pytypes.h>

std::vector<size_t> NMS_SIMD_PY::createRectangleIndices(const pybind11::list& scores, float scoreThreshold)
{
	std::vector<size_t> passRectIndices;
	for(size_t i = 0; i < scores.size(); i++)
	{
		if(scores[i].cast<float>() >= scoreThreshold)
		{
			passRectIndices.push_back(i);
		}
	}

	std::sort(passRectIndices.begin(), passRectIndices.end(), [&](size_t a, size_t b) {
		return scores[a] > scores[b];
	});

	return passRectIndices;
}

pybind11::list NMS_SIMD_PY::getValidIndices(const NMS_SIMD::Rectangles& rectangles, const std::vector<size_t>& indices)
{
	pybind11::list validIndices;
	for(size_t i = 0; i < rectangles.count; i++)
	{
		if(rectangles.validness[i] != 0)
		{
			validIndices.append(indices[i]);
		}
	}

	return validIndices;
}

template<class T>
void NMS_SIMD_PY::copyBufferToRectangles(T* buffer, NMS_SIMD::Rectangles rectangles, const std::vector<size_t>& indices)
{
	for(size_t i = 0; i < rectangles.count; i++)
	{
		size_t rectIdx = indices[i] * 4;
		rectangles.x1[i] = static_cast<uint32_t>(buffer[rectIdx]);
		rectangles.y1[i] = static_cast<uint32_t>(buffer[rectIdx + 1]);
		rectangles.x2[i] = static_cast<uint32_t>(buffer[rectIdx + 2]);
		rectangles.y2[i] = static_cast<uint32_t>(buffer[rectIdx + 3]);
	}
}

std::vector<size_t> NMS_SIMD_PY::createRectangleIndices(const pybind11::buffer& scores, float scoreThreshold)
{
	auto scoreBufferReq = scores.request();
	if(scoreBufferReq.ndim != 1)
	{
		throw std::runtime_error("Score buffer dimension should be 1.");
	}
	if(scoreBufferReq.format != pybind11::format_descriptor<float>::format())
	{
		throw pybind11::type_error("Score buffer should be np.float32");
	}

	float* buffer = reinterpret_cast<float*>(scoreBufferReq.ptr);
	std::vector<size_t> passRectIndices;
	for(size_t i = 0; i < scoreBufferReq.shape[0]; i++)
	{
		if(buffer[i] >= scoreThreshold)
		{
			passRectIndices.push_back(i);
		}
	}

	std::sort(passRectIndices.begin(), passRectIndices.end(), [&](size_t a, size_t b) {
		return buffer[a] > buffer[b];
	});

	return passRectIndices;
}

std::vector<size_t> NMS_SIMD_PY::runCreateRectangleIndices(pybind11::object scores, float scoreThreshold)
{
	if(pybind11::isinstance<pybind11::list>(scores))
	{
		return createRectangleIndices(scores.cast<pybind11::list>(), scoreThreshold);
	}
	else if(pybind11::isinstance<pybind11::buffer>(scores))
	{
		return createRectangleIndices(scores.cast<pybind11::buffer>(), scoreThreshold);
	}
	else
	{
		throw pybind11::type_error("Unknown type for scores, expected either list or ndarray.");
	}
}

void NMS_SIMD_PY::copyBufferToRectangles(pybind11::list buffer,
										 NMS_SIMD::Rectangles rectangles,
										 const std::vector<size_t>& indices)
{
	for(size_t i = 0; i < indices.size(); i++)
	{
		size_t rectIdx = indices[i];
		auto rect = buffer[rectIdx].cast<pybind11::list>();
		rectangles.x1[i] = rect[0].cast<uint32_t>();
		rectangles.y1[i] = rect[1].cast<uint32_t>();
		rectangles.x2[i] = rect[2].cast<uint32_t>();
		rectangles.y2[i] = rect[3].cast<uint32_t>();
	}

}

void NMS_SIMD_PY::runCopyBufferToRectangles(pybind11::object buffer,
											NMS_SIMD::Rectangles rectangles,
											const std::vector<size_t>& indices)
{
	if(pybind11::isinstance<pybind11::list>(buffer))
	{
		return copyBufferToRectangles(buffer.cast<py::list>(), rectangles, indices);
	}
	else if(pybind11::isinstance<pybind11::buffer>(buffer))
	{
		auto allRectangles = buffer.cast<py::buffer>();
		auto rectangleBufferRequest = allRectangles.request();
		assert(rectangleBufferRequest.ndim == 2);
		assert(rectangleBufferRequest.shape[1] == 4);
		if(rectangleBufferRequest.format == py::format_descriptor<float>::format())
		{
			copyBufferToRectangles(reinterpret_cast<float*>(rectangleBufferRequest.ptr), rectangles, indices);
		}
		else if(rectangleBufferRequest.format == py::format_descriptor<uint32_t>::format())
		{
			copyBufferToRectangles(reinterpret_cast<uint32_t*>(rectangleBufferRequest.ptr), rectangles, indices);
		}
		else if(rectangleBufferRequest.format == py::format_descriptor<int32_t>::format())
		{
			copyBufferToRectangles(reinterpret_cast<int32_t*>(rectangleBufferRequest.ptr), rectangles, indices);
		}
		else
		{
			throw py::type_error("Only valid buffer formats are np.float32, np.int32 and np.uint32");
		}
	}
	else
	{
		throw pybind11::type_error("Unknown type for all_rectangles, expected either list or ndarray.");
	}

}
