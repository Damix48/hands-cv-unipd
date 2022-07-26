#include "saver.h"

#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "hand.h"
#include "image.h"

Saver::Saver(std::string outputPath_) : outputPath(outputPath_) {}

void Saver::save(const std::vector<Image>& images) {
  std::filesystem::path detectionPath = outputPath;
  detectionPath.append("detected");
  std::filesystem::path maskPath = outputPath;
  maskPath.append("masks");
  std::filesystem::path boxPath = outputPath;
  boxPath.append("boxes");

  std::filesystem::create_directory(outputPath);
  std::filesystem::create_directory(detectionPath);
  std::filesystem::create_directory(maskPath);
  std::filesystem::create_directory(boxPath);

  for (int i = 0; i < images.size(); ++i) {
    std::string stem = std::filesystem::path(images[i].getPath()).stem();

    std::filesystem::path detectionOutput = detectionPath.append(stem + ".jpg");
    cv::imwrite(detectionOutput, images[i].getDetected());

    std::filesystem::path maskOutput = maskPath.append(stem + ".png");
    cv::imwrite(maskOutput, images[i].getMasks());

    std::filesystem::path boxOutput = boxPath.append(stem + ".txt");
    std::ofstream boxes(boxOutput);
    for (Hand hand : images[i].getHands()) {
      cv::Rect box = hand.getBox().toRect(images[i].size());
      std::string content = std::to_string(box.x) + "\t" + std::to_string(box.y) + "\t" + std::to_string(box.width) + "\t" + std::to_string(box.height) + "\n";

      boxes << content;
    }
    boxes.close();
  }
}