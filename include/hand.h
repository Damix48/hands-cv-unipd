#ifndef HAND_H
#define HAND_H

#include <opencv2/core.hpp>

#include "normalized_box.h"

class Hand {
  cv::Mat mask;

 public:
  NormalizedBox box;
  Hand(NormalizedBox box_);
  void prova();
};

#endif  // HAND_H