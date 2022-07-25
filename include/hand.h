#ifndef HAND_H
#define HAND_H

#include <opencv2/core.hpp>

#include "normalized_box.h"

class Hand {
  NormalizedBox box;
  cv::Mat mask;

 public:
  Hand(NormalizedBox box_);
  NormalizedBox getBox() const;

  float computeBoxIOU(Hand hand, cv::Size size);

  friend bool operator<(const Hand& left, const Hand& right);
  friend bool operator>(const Hand& left, const Hand& right);
  friend bool operator<=(const Hand& left, const Hand& right);
  friend bool operator>=(const Hand& left, const Hand& right);
};

#endif  // HAND_H