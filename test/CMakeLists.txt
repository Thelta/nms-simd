enable_testing()

project(nms_test)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(GTest CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_path(PCG_INCLUDE_DIRS "pcg_extras.hpp")
find_package(pybind11 CONFIG REQUIRED)

add_executable(
	${PROJECT_NAME}
	test.cpp
	native-test.h
	py-test.h
	${PCG_INCLUDE_DIRS}/pcg_extras.hpp
	${PCG_INCLUDE_DIRS}/pcg_random.hpp
)

target_link_libraries(
	${PROJECT_NAME}
	PRIVATE GTest::gtest GTest::gtest_main 
			${OpenCV_LIBS}
			nms-simd
			py_utils
			pybind11::embed
)

target_include_directories(
	${PROJECT_NAME}
	PRIVATE ${OpenCV_INCLUDE_LIBS}
			"../nms-simd"
			${PCG_INCLUDE_DIRS}
			"../python"
)

#target_link_options(${PROJECT_NAME} PRIVATE /fsanitize=address)
#target_compile_options(${PROJECT_NAME} PRIVATE /fsanitize=address)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
