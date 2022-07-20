#include "normalized_box.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <stdexcept>

NormalizedBox::NormalizedBox(float cx_, float cy_, float w_, float h_) {
  if (cx_ > 1 || cx_ < 0) {
    throw std::invalid_argument("Normalized center x must be in range [0, 1], is: " + std::to_string(cx_));
  }
  if (cy_ > 1 || cy_ < 0) {
    throw std::invalid_argument("Normalized center y must be in range [0, 1], is: " + std::to_string(cy_));
  }
  if (w_ > 1 || w_ < 0) {
    throw std::invalid_argument("Normalized width must be in range [0, 1], is: " + std::to_string(w_));
  }
  if (h_ > 1 || h_ < 0) {
    throw std::invalid_argument("Normalized height must be in range [0, 1], is: " + std::to_string(h_));
  }

  cx = cx_;
  cy = cy_;
  w = w_;
  h = h_;
}

NormalizedBox NormalizedBox::fromYolo(float yoloX_, float yoloY_, float yoloW_, float yoloH_, cv::Size size) {
  float cx_ = yoloX_ / size.width;
  float cy_ = yoloY_ / size.height;
  float w_ = yoloW_ / size.width;
  float h_ = yoloH_ / size.height;

  return NormalizedBox(cx_, cy_, w_, h_);
}

NormalizedBox NormalizedBox::fromRect(cv::Rect rect, cv::Size size) {
  float cx_ = (rect.x + (rect.width / 2)) / size.width;
  float cy_ = (rect.y + (rect.height / 2)) / size.height;
  float w_ = rect.width / size.width;
  float h_ = rect.height / size.height;

  return NormalizedBox(cx_, cy_, w_, h_);
}

cv::Rect NormalizedBox::toRect(cv::Size size) const {
  float x_ = (cx * size.width) - (w * size.width / 2);
  float y_ = (cy * size.height) - (h * size.height / 2);
  float w_ = w * size.width;
  float h_ = h * size.height;

  return cv::Rect(x_, y_, w_, h_);
}