#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

#include "image.h"
#include "yolo_detector.h"

int main(int argc, const char** argv) {
  const std::string keys =
      "{help h usage ? |      | print this message   }"
      "{@path          |      | image path           }"
      "{@model         |      | yolo model path      }"
      "{folder         |      | image path           }";

  cv::CommandLineParser parser(argc, argv, keys);

  parser.about("Application name v1.0.0");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  std::string path = parser.get<std::string>("@path");
  std::string modelPath = parser.get<std::string>("@model");
  std::string folder = parser.get<std::string>("folder");

  YoloDetector detector(modelPath, 640);

  std::vector<std::string> files;

  cv::glob(folder, files);

  for (int i = 0; i < files.size(); i++) {
    path = files[i];

    Image img(path);

    detector.detect(img);

    cv::imshow("prova" + std::to_string(i), img.getDetected());

    cv::waitKey(0);
  }
}