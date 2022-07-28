#include <chrono>
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

int main(int argc, const char **argv) {
  const std::string keys =
      "{help h usage ? |       | print help message   }"
      "{@path          |       | image path           }"
      "{@model         |       | yolo model path                               }"
      "{boxes          |       | ground-truth bounding boxes path              }"
      "{masks          |       | ground-truth masks path           }"
      "{output         |       | output path          }"
      "{display        | false | show images during detection and segmentation }"
      "{timing         | false | show timings }";

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
  bool display = parser.get<bool>("display");
  bool timing = parser.get<bool>("timing");

  if (path == "" || modelPath == "") {
    parser.printMessage();
    return EXIT_FAILURE;
  }

  YoloDetector detector(modelPath, 640);
  Saver saver(outputPath);

  std::vector<Image> images = Loader::loadImages(path);

  if (boxesPath != "") {
    Loader::loadBoxes(boxesPath, images);
  }

  if (masksPath != "") {
    Loader::loadMasks(masksPath, images);
  }

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  for (int i = 0; i < images.size(); i++) {
    begin = std::chrono::steady_clock::now();

    Image &img = images[i];

    std::cout << "Image: " << img.getPath() << std::endl;

    std::chrono::steady_clock::time_point beginDetection = std::chrono::steady_clock::now();

    std::cout << "Performing detection, please wait..." << std::endl;
    detector.detect(img);

    std::chrono::steady_clock::time_point endDetection = std::chrono::steady_clock::now();
    if (timing) {
      std::cout << "Detection = " << std::chrono::duration_cast<std::chrono::milliseconds>(endDetection - beginDetection).count() << "ms" << std::endl;
    }

    if (display) {
      cv::imshow("Detection", img.getDetected());
      cv::waitKey(1);
    }

    std::chrono::steady_clock::time_point beginSegmentation = std::chrono::steady_clock::now();

    std::cout << "Performing segmentation, please wait..." << std::endl;
    img.generateMasks();

    std::chrono::steady_clock::time_point endSegmentation = std::chrono::steady_clock::now();
    if (timing) {
      std::cout << "Segmentation = " << std::chrono::duration_cast<std::chrono::milliseconds>(endSegmentation - beginSegmentation).count() << "ms" << std::endl;
    }

    end = std::chrono::steady_clock::now();
    if (timing) {
      std::cout << "Total = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl
                << std::endl;
    }

    Printer::print(img);

    if (display) {
      cv::imshow("Segmentation", img.getOverlayMasks());
      std::cout << "Press any key to show the next image..." << std::endl
                << std::endl
                << std::endl;

      cv::waitKey(0);
    }
  }

  saver.save(images);
}