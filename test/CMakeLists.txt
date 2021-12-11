enable_testing()

project(nms_test)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if(NMS_SIMD_BUILD_PYTHON)
	find_package(pybind11 CONFIG REQUIRED)
	set(PYTHON_TEST_SRC py-test.h)
	set(PYTHON_TEST_LINK_LIBS 
		py_utils 
		pybind11::embed
	)
	set(PYTHON_TEST_INCLUDE_FOLDERS "../python")
	add_definitions(-DBUILD_PYTHON_TESTS)
endif()

find_package(GTest CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_path(PCG_INCLUDE_DIRS "pcg_extras.hpp")

add_executable(
	${PROJECT_NAME}
	test.cpp
	native-test.h
	${PYTHON_TEST_SRC}
	${PCG_INCLUDE_DIRS}/pcg_extras.hpp
	${PCG_INCLUDE_DIRS}/pcg_random.hpp
)

target_link_libraries(
	${PROJECT_NAME}
	PRIVATE GTest::gtest GTest::gtest_main 
			${OpenCV_LIBS}
			nms-simd
			${PYTHON_TEST_LINK_LIBS}
)

target_include_directories(
	${PROJECT_NAME}
	PRIVATE ${OpenCV_INCLUDE_LIBS}
			"../nms-simd"
			${PCG_INCLUDE_DIRS}
			${PYTHON_TEST_INCLUDE_FOLDERS}
)

#target_link_options(${PROJECT_NAME} PRIVATE /fsanitize=address)
#target_compile_options(${PROJECT_NAME} PRIVATE /fsanitize=address)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
