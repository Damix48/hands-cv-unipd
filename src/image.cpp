#include "image.h"

#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <string>

#include "hand.h"

Image::Image(std::string path_) : path(path_) {
  data = cv::imread(path, cv::IMREAD_UNCHANGED);
}

const cv::Mat Image::getImageBlob(cv::Size size) const {
  cv::Mat bgr;
  if (data.channels() == 1) {
    cv::cvtColor(data, bgr, cv::COLOR_GRAY2BGR);
  } else {
    data.copyTo(bgr);
  }

  cv::Mat resized(bgr);
  cv::resize(bgr, resized, size);

  cv::Mat blob(data);
  cv::dnn::blobFromImage(resized, blob, 1 / 255.0f, size, cv::Scalar(0, 0, 0), true, false);

  return blob;
}

void Image::addDetectedHand(Hand hand) {
  detectedHands.push_back(hand);
}

cv::Mat Image::getDetected() const {
  cv::Mat temp;
  data.copyTo(temp);

  for (Hand hand : detectedHands) {
    cv::rectangle(temp, hand.getBox().toRect(data.size()), cv::Scalar(255, 0, 0));
  }

  return temp;
}

void Image::addGroundTruthHand(Hand hand) {
  groundTruthHands.push_back(hand);
}

void Image::generateMasks() {
  detectedMasks = cv::Mat::zeros(size(), CV_8U);

  for (Hand &hand : detectedHands) {
    hand.generateMask(data);

    cv::Mat roiMasks = detectedMasks(hand.getBox().toRect(detectedMasks.size()));

    hand.getMask().copyTo(roiMasks, hand.getMask());
  }
}

cv::Mat Image::getMasks() const {
  return detectedMasks;
}

cv::Mat Image::getOverlayMasks() const {
  cv::Mat result1;
  data.copyTo(result1);
  cv::Mat result;
  cv::cvtColor(result1, result, cv::COLOR_BGR2BGRA);
  int a = 0.7;
  std::vector<cv::Scalar> colors = {cv::Scalar(77, 62, 200, a), cv::Scalar(255, 138, 0, a), cv::Scalar(255, 82, 173, a), cv::Scalar(121, 164, 0, a)};

  for (int i = 0; i < detectedHands.size(); ++i) {
    const Hand &hand = detectedHands[i];
    cv::Mat mask = cv::Mat(hand.getMask().size(), CV_8UC4, cv::Scalar(0, 0, 0, 0));

    mask.setTo(colors[i % 4], hand.getMask());

    cv::Mat roiMasks = result(hand.getBox().toRect(result.size()));
    cv::add(roiMasks, mask, roiMasks);
  }

  return result;
}

void Image::setGroundTruthMasks(cv::Mat mask) {
  mask.copyTo(groundTruthMasks);
}

void Image::setGroundTruthMasks(std::string path) {
  groundTruthMasks = cv::imread(path, cv::IMREAD_GRAYSCALE);
}

std::vector<Hand> Image::getHands() const {
  return detectedHands;
}

std::string Image::getPath() const {
  return path;
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

float Image::getMasksAccuracy() const {
  int groundTruthPixels = groundTruthMasks.rows * groundTruthMasks.cols;

  cv::Mat correctMasks, groundTruthNotMasks, detectedNotMasks, correctNotMasks;

  cv::bitwise_and(groundTruthMasks, detectedMasks, correctMasks);

  cv::bitwise_not(groundTruthMasks, groundTruthNotMasks);
  cv::bitwise_not(detectedMasks, detectedNotMasks);
  cv::bitwise_and(groundTruthNotMasks, detectedNotMasks, correctNotMasks);

  int correctMasksPixels = cv::countNonZero(correctMasks);
  int correctNotMasksPixels = cv::countNonZero(correctNotMasks);

  return ((float)correctMasksPixels + (float)correctNotMasksPixels) / (float)groundTruthPixels;
}

cv::Size Image::size() const {
  return data.size();
}
