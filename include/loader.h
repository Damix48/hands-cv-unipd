#ifndef LOADER_H
#define LOADER_H

#include <filesystem>
#include <vector>

#include "hand.h"
#include "image.h"

class Loader {
 public:
  static std::vector<Image> loadImages(std::filesystem::path path_);
  static void loadBoxes(std::filesystem::path path_, std::vector<Image>& images);
  // static std::vector<Image> loadBoxes(std::filesystem::path path_);
};

#endif  // LOADER_H