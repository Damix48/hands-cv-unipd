#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/core.hpp>
#include <vector>

#include "hand.h"

class Image {
  cv::Mat data;
  std::vector<Hand> hands;

 public:
  Image(cv::Mat src);

  const cv::Mat getImageBlob(cv::Size size) const;

  cv::Size size() const;

  void setHands(std::vector<Hand> hands);
  std::vector<Hand>& getHands();
};

#endif  // IMAGE_H