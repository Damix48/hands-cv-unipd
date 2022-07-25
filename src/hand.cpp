#include "hand.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "normalized_box.h"

Hand::Hand(NormalizedBox box_) : box(box_) {}

NormalizedBox Hand::getBox() const {
  return box;
}

float Hand::computeBoxIOU(Hand hand, cv::Size size) {
  cv::Mat detectedBox = cv::Mat::zeros(size, CV_8U);
  cv::Mat groundTruthBox = cv::Mat::zeros(size, CV_8U);

  cv::Mat intersectionImg = cv::Mat::zeros(size, CV_8U);
  cv::Mat unionImg = cv::Mat::zeros(size, CV_8U);

  cv::rectangle(detectedBox, getBox().toRect(size), cv::Scalar(255), cv::FILLED);
  cv::rectangle(groundTruthBox, hand.getBox().toRect(size), cv::Scalar(255), cv::FILLED);

  cv::bitwise_and(detectedBox, groundTruthBox, intersectionImg);
  cv::bitwise_or(detectedBox, groundTruthBox, unionImg);

  // cv::imshow("detected", detectedBox);
  // cv::imshow("ground", groundTruthBox);
  // cv::imshow("intersection", intersectionImg);
  // cv::imshow("union", unionImg);

  // cv::waitKey(0);

  return (float)cv::countNonZero(intersectionImg) / (float)cv::countNonZero(unionImg);
}

bool operator<(const Hand& left, const Hand& right) {
  return left.getBox() < right.getBox();
}

bool operator>(const Hand& left, const Hand& right) {
  return right < left;
}

bool operator<=(const Hand& left, const Hand& right) {
  return !(left > right);
}

bool operator>=(const Hand& left, const Hand& right) {
  return !(left < right);
}