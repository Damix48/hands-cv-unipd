#include "saver.h"

#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "hand.h"
#include "image.h"

Saver::Saver(std::string outputPath_) : outputPath(outputPath_) {}

void Saver::save(const std::vector<Image>& images) {
  std::string detectionPath = cv::utils::fs::join(outputPath, "detected");
  std::string maskPath = cv::utils::fs::join(outputPath, "masks");
  std::string overlayPath = cv::utils::fs::join(outputPath, "overlay");
  std::string boxPath = cv::utils::fs::join(outputPath, "boxes");

  cv::utils::fs::createDirectory(outputPath);
  cv::utils::fs::createDirectory(detectionPath);
  cv::utils::fs::createDirectory(maskPath);
  cv::utils::fs::createDirectory(overlayPath);
  cv::utils::fs::createDirectory(boxPath);

  for (int i = 0; i < images.size(); ++i) {
    std::string imagePath = images[i].getPath();
    std::string stem = imagePath.substr(imagePath.size() - 6, 2);

    std::string detectionOutput = cv::utils::fs::join(detectionPath, stem + ".jpg");
    cv::imwrite(detectionOutput, images[i].getDetected());

    std::string maskOutput = cv::utils::fs::join(maskPath, stem + ".png");
    cv::imwrite(maskOutput, images[i].getMasks());

    std::string overlayOutput = cv::utils::fs::join(overlayPath, stem + ".jpg");
    cv::imwrite(overlayOutput, images[i].getOverlayMasks());

    std::string boxOutput = cv::utils::fs::join(boxPath, stem + ".txt");
    std::ofstream boxes(boxOutput);
    for (Hand hand : images[i].getHands()) {
      cv::Rect box = hand.getBox().toRect(images[i].size());
      std::string content = std::to_string(box.x) + "\t" + std::to_string(box.y) + "\t" + std::to_string(box.width) + "\t" + std::to_string(box.height) + "\n";

      boxes << content;
    }
    boxes.close();
  }
}