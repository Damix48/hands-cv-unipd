#ifndef LOADER_H
#define LOADER_H

#include <string>
#include <vector>

#include "hand.h"
#include "image.h"

class Loader {
 public:
  static std::vector<Image> loadImages(std::string path_);
  static void loadBoxes(std::string path_, std::vector<Image>& images);
  static void loadMasks(std::string path_, std::vector<Image>& images);
};

#endif  // LOADER_H