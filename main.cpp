#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

#include "image.h"
#include "loader.h"
#include "printer.h"
#include "saver.h"
#include "yolo_detector.h"

int main(int argc, const char** argv) {
  const std::string keys =
      "{help h usage ? |      | print this message   }"
      "{@path          |      | image path           }"
      "{@model         |      | yolo model path      }"
      "{boxes          |      | bounding boxes path  }"
      "{masks          |      | masks path           }"
      "{output         |      | output path          }";

  cv::CommandLineParser parser(argc, argv, keys);

  parser.about("Hand v1.0.0");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  std::string path = parser.get<std::string>("@path");
  std::string modelPath = parser.get<std::string>("@model");
  std::string boxesPath = parser.get<std::string>("boxes");
  std::string masksPath = parser.get<std::string>("masks");
  std::string outputPath = parser.get<std::string>("output");

  YoloDetector detector(modelPath, 640);
  Saver saver(outputPath);

  std::vector<Image> images = Loader::loadImages(path);

  if (boxesPath != "") {
    Loader::loadBoxes(boxesPath, images);
  }

  if (masksPath != "") {
    Loader::loadMasks(masksPath, images);
  }

  for (int i = 0; i < images.size(); i++) {
    Image& img = images[i];

    detector.detect(img);

    // cv::imshow("prova" + std::to_string(i), img.getDetected());
    // cv::imwrite(outputPath + "//prova" + std::to_string(i) + ".jpg", img.getDetected());
    // cv::waitKey(0);

    img.generateMasks();

    Printer::print(img);

    // cv::imshow("mask" + std::to_string(i), img.getMasks());
  }

  saver.save(images);
}