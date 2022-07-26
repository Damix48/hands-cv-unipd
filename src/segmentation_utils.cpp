#include "segmentation_utils.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat segmentation::grabCutRect(cv::Mat src, int iter, int padding) {
  cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);

  cv::Rect box = cv::Rect(padding, padding, src.cols - 2 * padding, src.rows - 2 * padding);

  mask.setTo(cv::GC_PR_BGD);
  (mask(box)).setTo(cv::Scalar(cv::GC_FGD));

  // cv::imshow("fra", src);
  // cv::imshow("fra2", mask * 60);
  // cv::waitKey(0);

  cv::Mat bgdModel;
  cv::Mat fgdModel;
  cv::grabCut(src, mask, box, bgdModel, fgdModel, iter, cv::GC_INIT_WITH_RECT);

  cv::Mat dst;
  cv::threshold(mask, dst, cv::GC_PR_FGD - 1, 255, cv::THRESH_BINARY);

  return dst;
}
