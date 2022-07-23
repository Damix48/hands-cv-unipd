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
  float cx_ = (float)(rect.x + (rect.width / 2)) / (float)size.width;
  float cy_ = (float)(rect.y + (rect.height / 2)) / (float)size.height;
  float w_ = (float)rect.width / (float)size.width;
  float h_ = (float)rect.height / (float)size.height;

  return NormalizedBox(cx_, cy_, w_, h_);
}

cv::Rect NormalizedBox::toRect(cv::Size size) const {
  float x_ = (cx * size.width) - (w * size.width / 2);
  float y_ = (cy * size.height) - (h * size.height / 2);
  float w_ = w * size.width;
  float h_ = h * size.height;

  return cv::Rect(x_, y_, w_, h_);
}

bool operator<(const NormalizedBox& left, const NormalizedBox& right) {
  if (left.cx < right.cx) {
    return true;
  }

  if (left.cx > right.cx) {
    return false;
  }

  if (left.cy < right.cy) {
    return true;
  }

  if (left.cy > right.cy) {
    return false;
  }

  if (left.w < right.w) {
    return true;
  }

  if (left.w > right.w) {
    return false;
  }

  if (left.h < right.h) {
    return true;
  }

  if (left.h > right.h) {
    return false;
  }

  return false;
}

bool operator>(const NormalizedBox& left, const NormalizedBox& right) {
  return right < left;
}

bool operator<=(const NormalizedBox& left, const NormalizedBox& right) {
  return !(left > right);
}

bool operator>=(const NormalizedBox& left, const NormalizedBox& right) {
  return !(left < right);
}

bool operator==(const NormalizedBox& left, const NormalizedBox& right) {
  return (left.cx == right.cx) && (left.cy == right.cy) && (left.w == right.w) && (left.h == right.h);
}

bool operator!=(const NormalizedBox& left, const NormalizedBox& right) {
  return !(left == right);
}