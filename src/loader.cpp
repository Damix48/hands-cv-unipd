#include "loader.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core/utils/filesystem.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "hand.h"
#include "image.h"
#include "normalized_box.h"

std::vector<Image> Loader::loadImages(std::filesystem::path path_) {
  if (!std::filesystem::exists(path_)) {
    throw std::invalid_argument("The path specified (" + path_.string() + ") don't exists");
  }

  std::vector<std::string> paths;

  if (std::filesystem::is_directory(path_)) {
    cv::glob(path_, paths);
  } else {
    paths.push_back(path_);
  }

  std::vector<Image> images;

  for (std::filesystem::path path : paths) {
    images.push_back(Image(path));
  }

  return images;
}

void Loader::loadBoxes(std::filesystem::path path_, std::vector<Image>& images) {
  if (!std::filesystem::exists(path_)) {
    throw std::invalid_argument("The path specified (" + path_.string() + ") don't exists");
  }

  std::vector<std::string> paths;

  if (std::filesystem::is_directory(path_)) {
    cv::glob(path_, paths);
  } else {
    paths.push_back(path_);
  }

  if (paths.size() != images.size()) {
    throw std::invalid_argument("The number of boxes files (" + std::to_string(paths.size()) + ") is different to the number of images (" + std::to_string(images.size()) + ")");
  }

  for (int i = 0; i < paths.size(); ++i) {
    std::filesystem::path path = paths[i];
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

void Loader::loadMasks(std::filesystem::path path_, std::vector<Image>& images) {
  if (!std::filesystem::exists(path_)) {
    throw std::invalid_argument("The path specified (" + path_.string() + ") don't exists");
  }

  std::vector<std::string> paths;

  if (std::filesystem::is_directory(path_)) {
    cv::glob(path_, paths);
  } else {
    paths.push_back(path_);
  }

  if (paths.size() != images.size()) {
    throw std::invalid_argument("The number of boxes files (" + std::to_string(paths.size()) + ") is different to the number of images (" + std::to_string(images.size()) + ")");
  }

  for (int i = 0; i < paths.size(); ++i) {
    std::filesystem::path path = paths[i];
    Image& image = images[i];

    image.setGroundTruthMasks(path);
  }
}