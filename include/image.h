#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "hand.h"

class Image {
  cv::Mat data;

  std::vector<Hand> detectedHands;
  std::vector<Hand> groundTruthHands;

 public:
  Image(cv::Mat src);
  Image(std::string path);

  const cv::Mat getImageBlob(cv::Size size) const;

  void addDetectedHand(Hand hand);
  void addGroundTruthHand(Hand hand);

  cv::Mat getDetected() const;

  std::vector<float> getIOUs() const;

  cv::Size size() const;
};

#endif  // IMAGE_H