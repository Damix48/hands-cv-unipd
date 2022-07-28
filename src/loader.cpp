#include "loader.h"

#include <fstream>
#include <opencv2/core/utils/filesystem.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "hand.h"
#include "image.h"
#include "normalized_box.h"

std::vector<Image> Loader::loadImages(std::string path_) {
  if (!cv::utils::fs::exists(path_)) {
    throw std::invalid_argument("The path specified (" + path_ + ") don't exists");
  }

  std::vector<std::string> paths;

  cv::glob(path_, paths);

  std::vector<Image> images;

  for (std::string path : paths) {
    images.push_back(Image(path));
  }

  return images;
}

void Loader::loadBoxes(std::string path_, std::vector<Image>& images) {
  if (!cv::utils::fs::exists(path_)) {
    throw std::invalid_argument("The path specified (" + path_ + ") don't exists");
  }

  std::vector<std::string> paths;

  cv::glob(path_, paths);

  if (paths.size() != images.size()) {
    throw std::invalid_argument("The number of boxes files (" + std::to_string(paths.size()) + ") is different to the number of images (" + std::to_string(images.size()) + ")");
  }

  for (int i = 0; i < paths.size(); ++i) {
    std::string path = paths[i];
    Image& image = images[i];

    std::vector<Hand> hands_;

    std::ifstream file(path);

    std::vector<int> data;
    int f;
    while (file >> f) {
      data.push_back(f);
    }

    for (int j = 0; j < data.size(); j += 4) {
      int x = data[j];
      int y = data[j + 1];
      int w = data[j + 2];
      int h = data[j + 3];

      NormalizedBox box = NormalizedBox::fromRect(cv::Rect(x, y, w, h), image.size());

      image.addGroundTruthHand(Hand(box));
    }
  }
}

void Loader::loadMasks(std::string path_, std::vector<Image>& images) {
  if (!cv::utils::fs::exists(path_)) {
    throw std::invalid_argument("The path specified (" + path_ + ") don't exists");
  }

  std::vector<std::string> paths;

  cv::glob(path_, paths);

  if (paths.size() != images.size()) {
    throw std::invalid_argument("The number of masks files (" + std::to_string(paths.size()) + ") is different to the number of images (" + std::to_string(images.size()) + ")");
  }

  for (int i = 0; i < paths.size(); ++i) {
    std::string path = paths[i];
    Image& image = images[i];

    image.setGroundTruthMasks(path);
  }
}