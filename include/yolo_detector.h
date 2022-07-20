#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <opencv2/dnn.hpp>
#include <string>

#include "image.h"

class YoloDetector {
  cv::dnn::Net model;
  cv::Size inputSize;
  std::vector<std::string> outputLayerNames;

 public:
  YoloDetector(std::string modelPath, int inputSize_ = 640);
  void detect(Image& img);
};

#endif  // YOLO_DETECTOR_H