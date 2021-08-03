enable_testing()

project(nms_test)

find_package(GTest CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_path(PCG_INCLUDE_DIRS "pcg_extras.hpp")

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
set(addressSanitizerEnabled true)


target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})