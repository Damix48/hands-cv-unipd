#ifndef SAVER_H
#define SAVER_H

#include <filesystem>
#include <vector>

#include "image.h"

class Saver {
  std::filesystem::path outputPath;

  // void saveImages(std::string path, std::vector<Image> images);

 public:
  Saver(std::string outputPath_);
  void save(const std::vector<Image>& images);
};

#endif  // SAVER_H