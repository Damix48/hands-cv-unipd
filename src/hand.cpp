#include "hand.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "normalized_box.h"
#include "segmentation_utils.h"

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

cv::Mat Hand::getHandBox(cv::Mat src, float scale, int padding) {
  cv::Mat handBox;

  cv::Rect rect = getBox().toRect(src.size());
  if (rect.x != 0) {
    rect.x -= padding;
  }
  if (rect.y != 0) {
    rect.y -= padding;
  }
  if (rect.x + rect.width + 2 * padding < src.cols) {
    rect.width += 2 * padding;
  }
  if (rect.y + rect.height + 2 * padding < src.rows) {
    rect.height += 2 * padding;
  }

  src(rect).copyTo(handBox);

  cv::Size sizeScaled(rect.width, rect.height);
  sizeScaled.width *= scale;
  sizeScaled.height *= scale;

  cv::resize(handBox, handBox, sizeScaled);

  return handBox;
}

void Hand::generateMask(cv::Mat src) {
  cv::Mat bgr;
  if (src.channels() == 1) {
    cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
  } else {
    src.copyTo(bgr);
  }

  int padding = 2;
  float scale = 2;

  cv::Mat img = getHandBox(bgr, scale, padding);

  cv::Mat grabCutMask = segmentation::grabCutRect(img, 10, padding);

  cv::Rect rect = getBox().toRect(bgr.size());
  cv::Size sizeScaled(rect.width, rect.height);

  cv::resize(grabCutMask, grabCutMask, sizeScaled);

  grabCutMask.copyTo(mask);
}

cv::Mat Hand::getMask() {
  return mask;
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

void Hand::showSkin(cv::Mat img) {
  cv::Mat hand;
  img.copyTo(hand);
  hand = hand(getBox().toRect(img.size()));

  cv::Mat hsv, hsv_mask;
  cv::cvtColor(hand, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv, cv::Scalar(0, 15, 0), cv::Scalar(17, 170, 255), hsv_mask);
  cv::morphologyEx(hsv_mask, hsv_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

  cv::Mat YCrCb, YCrCb_mask;
  cv::cvtColor(hand, YCrCb, cv::COLOR_BGR2YCrCb);
  cv::inRange(YCrCb, cv::Scalar(0, 135, 85), cv::Scalar(255, 180, 135), YCrCb_mask);
  cv::morphologyEx(YCrCb_mask, YCrCb_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

  cv::Mat mask;
  cv::bitwise_and(hsv_mask, YCrCb_mask, mask);
  cv::medianBlur(mask, mask, 3);
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4)));

  cv::imshow("hsv", hsv_mask);
  cv::imshow("hYCrCbsv", YCrCb_mask);
  cv::imshow("mask", mask);

  cv::waitKey(0);
}