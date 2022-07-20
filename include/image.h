#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "hand.h"

class Image {
  cv::Mat data;
  std::vector<Hand> hands;

 public:
  Image(cv::Mat src);
  Image(std::string path);

  const cv::Mat getImageBlob(cv::Size size) const;

  void addHand(Hand hand);

  cv::Mat getDetected() const;

  cv::Size size() const;

  std::vector<Hand>& getHands();
};

#endif  // IMAGE_H