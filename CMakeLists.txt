cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(HandDetection)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include/)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})