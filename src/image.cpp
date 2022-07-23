#include "image.h"

#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "hand.h"

Image::Image(cv::Mat src) : data(src) {}

Image::Image(std::string path) {
  data = cv::imread(path);
}

const cv::Mat Image::getImageBlob(cv::Size size) const {
  cv::Mat resized(data);
  cv::resize(data, resized, size);

  cv::Mat blob(data);
  cv::dnn::blobFromImage(resized, blob, 1 / 255.0f, size, cv::Scalar(0, 0, 0), true, false);

  return blob;
}

void Image::addDetectedHand(Hand hand) {
  detectedHands.push_back(hand);
}

void Image::addGroundTruthHand(Hand hand) {
  groundTruthHands.push_back(hand);
}

cv::Mat Image::getDetected() const {
  cv::Mat temp(data);
  for (Hand hand : detectedHands) {
    cv::rectangle(temp, hand.getBox().toRect(data.size()), cv::Scalar(255, 0, 0));
  }

  return temp;
}

std::vector<float> Image::getIOUs() const {
  std::vector<float> IOCs;

  std::vector<Hand> detected = detectedHands;
  std::vector<Hand> groundTruth = groundTruthHands;

  std::sort(detected.begin(), detected.end());
  std::sort(groundTruth.begin(), groundTruth.end());

  int size_ = detected.size();

  if (detected.size() > groundTruth.size()) {
    size_ = groundTruth.size();
  }

  for (int i = 0; i < size_; ++i) {
    IOCs.push_back(detected[i].computeBoxIOU(groundTruth[i], size()));
  }

  return IOCs;
}

cv::Size Image::size() const {
  return data.size();
}
