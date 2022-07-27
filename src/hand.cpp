#include "hand.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/intensity_transform.hpp>

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

cv::Mat Hand::getHandBox(cv::Mat src, float &scale, int padding) {
  cv::Mat handBox;

  cv::Rect rect = getBox().toRect(src.size());
  if (rect.x > padding) {
    rect.x -= padding;
  }
  if (rect.y > padding) {
    rect.y -= padding;
  }
  if (rect.x + rect.width + padding + 1 < src.cols) {
    rect.width += padding + 1;
  }
  if (rect.y + rect.height + padding + 1 < src.rows) {
    rect.height += padding + 1;
  }

  src(rect).copyTo(handBox);

  if (handBox.cols * scale > 600 || handBox.rows * scale > 600) {
    if (handBox.cols > handBox.rows) {
      int width = 600;
      scale = (float)width / (float)handBox.cols;
    } else {
      int height = 600;
      scale = (float)height / (float)handBox.rows;
    }
  }

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

  cv::Mat grabCutMask = segmentation::grabCutRect(img, 10, padding);  // choose 7 or 10

  int nonzeros = cv::countNonZero(grabCutMask);
  int i = 0;
  while (nonzeros < grabCutMask.rows * grabCutMask.cols * 0.1 && i < 3) {
    scale += 0.2;
    img = getHandBox(bgr, scale, padding);
    grabCutMask = segmentation::grabCutRect(img, 10, padding);
    nonzeros = cv::countNonZero(grabCutMask);
  }

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
  cv::morphologyEx(grabCutMask, grabCutMask, cv::MORPH_OPEN, kernel);

  // discharge the smallest (with area < 20%) connected component
  grabCutMask = segmentation::getLargestConnectedComponents(grabCutMask, 0.2 * grabCutMask.cols * grabCutMask.rows);

  cv::Mat img1 = segmentation::getMaskIntersectImage(img, grabCutMask);

  cv::intensity_transform::BIMEF(img1, img1);

  cv::Mat img2 = segmentation::SLICSuperPixel(img1, 300, 20, 30.0, 25, 100);

  cv::Mat threshMask = segmentation::skinThreshold(img2);
  cv::Mat img3 = segmentation::getMaskIntersectImage(img1, threshMask);

  // if thresholding is gone wrong (e.g., because of hands with gloves) keep grabCutMask as final result
  cv::Mat result;
  if (cv::countNonZero(threshMask) < 0.1 * threshMask.rows * threshMask.cols) {
    result = img1;
  } else {
    result = segmentation::getMaskIntersectImage(img1, threshMask);
  }

  // Convert the result of the segmentation in a mask of type CV_8UC1
  cv::Mat segmMask = cv::Mat::zeros(result.size(), CV_8UC1);
  for (int i = 0; i < result.rows; i++) {
    for (int j = 0; j < result.cols; j++) {
      if (result.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0)) {
        segmMask.at<uchar>(i, j) = 255;
      }
    }
  }

  cv::Rect rect = getBox().toRect(bgr.size());
  cv::Size sizeScaled(rect.width, rect.height);

  cv::resize(segmMask, segmMask, sizeScaled);

  segmMask.copyTo(mask);
}

cv::Mat Hand::getMask() const {
  return mask;
}

bool operator<(const Hand &left, const Hand &right) {
  return left.getBox() < right.getBox();
}

bool operator>(const Hand &left, const Hand &right) {
  return right < left;
}

bool operator<=(const Hand &left, const Hand &right) {
  return !(left > right);
}

bool operator>=(const Hand &left, const Hand &right) {
  return !(left < right);
}
