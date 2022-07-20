cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(HandDetection)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} include/)

add_library(normalized_box_lib src/normalized_box.cpp include/normalized_box.h)
add_library(hand_lib src/hand.cpp include/hand.h)
add_library(image_lib src/image.cpp include/image.h)
add_library(yolo_detector_lib src/yolo_detector.cpp include/yolo_detector.h)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(hand_lib normalized_box_lib)
target_link_libraries(image_lib hand_lib)
target_link_libraries(yolo_detector_lib image_lib hand_lib normalized_box_lib)
target_link_libraries(${PROJECT_NAME} yolo_detector_lib image_lib)