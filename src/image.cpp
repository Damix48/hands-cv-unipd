#include "image.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "hand.h"

Image::Image(cv::Mat src) : data(src) {}

const cv::Mat Image::getImageBlob(cv::Size size) const {
  cv::Mat resized;
  cv::Mat blob;

  cv::resize(data, resized, size);

  cv::dnn::blobFromImage(resized, blob, 1 / 255.0f, size, cv::Scalar(0, 0, 0), true, false);

  return blob;
}