#ifndef HAND_H
#define HAND_H

#include <opencv2/core.hpp>

class Hand {
  cv::Rect box;

 public:
  Hand();
};

#endif  // HAND_H