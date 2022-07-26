#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "hand.h"

class Image {
  std::string path;
  cv::Mat data;

  std::vector<Hand> detectedHands;
  std::vector<Hand> groundTruthHands;

  cv::Mat detectedMasks;
  cv::Mat groundTruthMasks;

 public:
  Image(std::string path);

  const cv::Mat getImageBlob(cv::Size size) const;

  void addDetectedHand(Hand hand);
  cv::Mat getDetected() const;
  void addGroundTruthHand(Hand hand);

  void generateMasks();
  cv::Mat getMasks() const;
  void setGroundTruthMasks(cv::Mat masks);
  void setGroundTruthMasks(std::string path);

  std::vector<Hand> getHands() const;

  std::string getPath() const;

  std::vector<float> getIOUs() const;
  float getMasksAccuracy() const;

  cv::Size size() const;
};

#endif  // IMAGE_H