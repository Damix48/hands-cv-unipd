#include "image.h"

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

void Image::addHand(Hand hand) {
  hands.push_back(hand);
}

cv::Mat Image::getDetected() const {
  cv::Mat temp(data);
  for (Hand hand : hands) {
    cv::rectangle(temp, hand.box.toRect(data.size()), cv::Scalar(255, 0, 0));
  }

  return temp;
}

cv::Size Image::size() const {
  return data.size();
}
