enable_testing()

project(nms_test)

find_package(GTest CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_path(PCG_INCLUDE_DIRS "pcg_extras.hpp")
find_package(pybind11 CONFIG REQUIRED)

add_executable(
	${PROJECT_NAME}
	test.cpp
	${PCG_INCLUDE_DIRS}/pcg_extras.hpp
	${PCG_INCLUDE_DIRS}/pcg_random.hpp
)

target_link_libraries(
	${PROJECT_NAME}
	PRIVATE GTest::gtest GTest::gtest_main 
			${OpenCV_LIBS}
			nms-simd
)

target_include_directories(
	${PROJECT_NAME}
	PRIVATE ${OpenCV_INCLUDE_LIBS}
			"../nms-simd"
			${PCG_INCLUDE_DIRS}
)

target_link_options(${PROJECT_NAME} PRIVATE /fsanitize=address)
target_compile_options(${PROJECT_NAME} PRIVATE /fsanitize=address)

add_executable(
	py_utils_test
	py-test.cpp
)

target_link_libraries(
	py_utils_test
	PRIVATE GTest::gtest GTest::gtest_main
	nms-simd
	py_utils
	pybind11::embed
)

target_include_directories(
	py_utils_test
	PRIVATE "../nms-simd"
			"../python"
)


target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
gtest_discover_tests(py_utils_test)