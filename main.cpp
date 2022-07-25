#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

#include "image.h"
#include "loader.h"
#include "yolo_detector.h"

int main(int argc, const char** argv) {
  const std::string keys =
      "{help h usage ? |      | print this message   }"
      "{@path          |      | image path           }"
      "{@model         |      | yolo model path      }"
      "{boxes          |      | bounding boxes path  }"
      "{masks          |      | masks path  }";

  cv::CommandLineParser parser(argc, argv, keys);

  parser.about("Application name v1.0.0");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  std::string path = parser.get<std::string>("@path");
  std::string modelPath = parser.get<std::string>("@model");
  std::string boxesPath = parser.get<std::string>("boxes");
  std::string masksPath = parser.get<std::string>("masks");

  YoloDetector detector(modelPath, 640);

  std::vector<Image> images = Loader::loadImages(path);

  if (boxesPath != "") {
    Loader::loadBoxes(boxesPath, images);
  }

  for (int i = 0; i < images.size(); i++) {
    Image img = images[i];

    detector.detect(img);

    cv::imshow("prova" + std::to_string(i), img.getDetected());

    for (float IOU : img.getIOUs()) {
      std::cout << IOU << std::endl;
    }

    cv::waitKey(0);
  }
}