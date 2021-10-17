project(nms_benchmark)

find_package(benchmark CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_path(PCG_INCLUDE_DIRS "pcg_extras.hpp")

add_executable(
	${PROJECT_NAME}
	benchmark.cpp
)

target_link_libraries(
	${PROJECT_NAME}
	PRIVATE benchmark::benchmark 
			${OpenCV_LIBS}
			nms-simd
)

target_include_directories(
	${PROJECT_NAME}
	PRIVATE ${OpenCV_INCLUDE_LIBS}
			"../nms-simd"
			${PCG_INCLUDE_DIRS}
)
