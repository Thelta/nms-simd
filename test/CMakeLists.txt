enable_testing()

project(nms_test)

add_executable(
	${PROJECT_NAME}
	test.cpp
)

find_package(GTest CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)

target_link_libraries(
	${PROJECT_NAME}
	PRIVATE GTest::gtest GTest::gtest_main 
			${OpenCV_LIBS}
			absl::random_random absl::random_distributions
)

target_include_directories(
	${PROJECT_NAME}
	PRIVATE ${OpenCV_INCLUDE_LIBS}
			"../nms-simd"
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})