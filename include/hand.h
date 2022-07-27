#ifndef HAND_H
#define HAND_H

#include <opencv2/core.hpp>

#include "normalized_box.h"

class Hand
{
  NormalizedBox box;
  cv::Mat mask;

  cv::Mat getHandBox(cv::Mat src, float &scale, int padding = 0);

public:
  Hand(NormalizedBox box_);
  NormalizedBox getBox() const;

  float computeBoxIOU(Hand hand, cv::Size size);

  void generateMask(cv::Mat src);
  cv::Mat getMask() const;

  void showSkin(cv::Mat img);

  friend bool operator<(const Hand &left, const Hand &right);
  friend bool operator>(const Hand &left, const Hand &right);
  friend bool operator<=(const Hand &left, const Hand &right);
  friend bool operator>=(const Hand &left, const Hand &right);
};

#endif // HAND_H