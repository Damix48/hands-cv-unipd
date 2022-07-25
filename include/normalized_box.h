#ifndef NORMALIZED_BOX_H
#define NORMALIZED_BOX_H

#include <opencv2/core.hpp>

class NormalizedBox {
  float cx;
  float cy;
  float w;
  float h;

 public:
  NormalizedBox(float cx_, float cy_, float w_, float h_);

  static NormalizedBox fromYolo(float yoloX_, float yoloY_, float yoloW_, float yoloH_, cv::Size size);
  static NormalizedBox fromRect(cv::Rect rect, cv::Size size);

  cv::Rect toRect(cv::Size size) const;

  friend bool operator<(const NormalizedBox& left, const NormalizedBox& right);
  friend bool operator>(const NormalizedBox& left, const NormalizedBox& right);
  friend bool operator<=(const NormalizedBox& left, const NormalizedBox& right);
  friend bool operator>=(const NormalizedBox& left, const NormalizedBox& right);
  friend bool operator==(const NormalizedBox& left, const NormalizedBox& right);
  friend bool operator!=(const NormalizedBox& left, const NormalizedBox& right);
};

#endif  // NORMALIZED_BOX_H