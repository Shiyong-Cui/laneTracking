
cmake_minimum_required(VERSION 3.0)

project(lane)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)

set(src_list 
	src/DashboardCamera.cpp
	src/lane.cpp
	src/LaneFinder.cpp
	src/Window.cpp
	src/WindowFilter.cpp	
)

add_executable(laneTracking ${src_list})
target_include_directories(laneTracking PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
target_link_libraries(laneTracking ${OpenCV_LIBS})
