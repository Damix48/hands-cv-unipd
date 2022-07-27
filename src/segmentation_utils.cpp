#include "segmentation_utils.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/slic.hpp>

cv::Mat segmentation::getLargestConnectedComponents(cv::Mat src, int minArea) {
  cv::Mat labels = cv::Mat::zeros(src.size(), CV_8U);
  cv::Mat stats;
  cv::Mat centroids;

  std::vector<int> components;

  cv::connectedComponentsWithStats(src, labels, stats, centroids, 4);

  for (int i = 0; i < stats.rows; ++i) {
    if (stats.at<int>(i, cv::CC_STAT_AREA) > minArea) {
      components.push_back(i);
    }
  }

  cv::Mat result = cv::Mat::zeros(src.size(), CV_8U);

  for (int index : components) {
    cv::Mat single;
    cv::inRange(labels, index, index, single);
    cv::bitwise_or(result, single, result);
  }

  cv::Mat dst;
  cv::bitwise_and(src, result, dst);
  return dst;
}

cv::Mat segmentation::grabCutRect(cv::Mat src, int iter, int padding) {
  cv::Mat mask = cv::Mat::zeros(src.size(), CV_8U);

  cv::Rect box = cv::Rect(padding, padding, src.cols - padding - 1, src.rows - padding - 1);

  mask.setTo(cv::GC_PR_BGD);
  (mask(box)).setTo(cv::Scalar(cv::GC_FGD));

  // cv::imshow("srcGC", src);
  // cv::imshow("maskGC", mask * 60);
  // cv::waitKey(0);

  cv::Mat bgdModel;
  cv::Mat fgdModel;
  cv::grabCut(src, mask, box, bgdModel, fgdModel, iter, cv::GC_INIT_WITH_RECT);

  cv::Mat dst;
  cv::threshold(mask, dst, cv::GC_PR_FGD - 1, 255, cv::THRESH_BINARY);

  return dst;
}

cv::Mat segmentation::SLICSuperPixel(cv::Mat src, int superpixelNumber, int regionSize, float ruler, int minElementSize, int numberIterations) {
  cv::Ptr<cv::ximgproc::SuperpixelSLIC> superpix = cv::ximgproc::createSuperpixelSLIC(src, cv::ximgproc::SLIC, regionSize, ruler);
  superpix->iterate(numberIterations);

  // merge small superpixels
  if (minElementSize > 0) {
    superpix->enforceLabelConnectivity(minElementSize);
  }
  int nPix = superpix->getNumberOfSuperpixels();

  cv::Mat labels;
  superpix->getLabels(labels);

  // visualization
  cv::Mat dst = cv::Mat::zeros(labels.size(), CV_8UC3);
  std::vector<cv::Vec3b> labelColor;
  std::vector<cv::Mat> channels;
  cv::split(src, channels);

  // compute the mean of the pixels colors for each superpixel
  for (int l = 0; l < nPix; l++) {
    int b = 0;
    int g = 0;
    int r = 0;
    int count = 0;
    for (int h = 0; h < labels.rows; h++) {
      for (int k = 0; k < labels.cols; k++) {
        if (labels.at<int>(h, k) == l) {
          count++;
          b += channels[0].at<uchar>(h, k);
          g += channels[1].at<uchar>(h, k);
          r += channels[2].at<uchar>(h, k);
        }
      }
    }
    b /= count;
    g /= count;
    r /= count;
    labelColor.push_back(cv::Vec3b(b, g, r));
  }

  // assign to each superpixel the color previously computed
  for (int l = 0; l < labelColor.size(); l++) {
    for (int h = 0; h < labels.rows; h++) {
      for (int k = 0; k < labels.cols; k++) {
        if (labels.at<int>(h, k) == l) {
          dst.at<cv::Vec3b>(h, k) = labelColor[l];
        }
      }
    }
  }
  return dst;
}

cv::Mat segmentation::skinThreshold(cv::Mat src) {
  cv::Mat srcHSV;
  std::vector<cv::Mat> HSVchannels;
  cv::cvtColor(src, srcHSV, cv::COLOR_BGR2HSV);
  cv::split(srcHSV, HSVchannels);

  // in our case the HSV space is sufficient
  // cv::Mat srcYCrCb;
  // std::vector<cv::Mat> YCrCbchannels;
  // cv::cvtColor(src, srcYCrCb, cv::COLOR_BGR2YCrCb);
  // cv::split(srcYCrCb, YCrCbchannels);

  // cv::Mat srcNormRGB;
  // cv::cvtColor(src, srcNormRGB, cv::COLOR_BGR2RGB);
  // std::vector<cv::Mat> RGBChannels;
  // cv::split(srcNormRGB, RGBChannels);
  // std::vector<cv::Mat> normRGBChannels = normalizeRGB(RGBChannels[0], RGBChannels[1], RGBChannels[2]);

  cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);
  for (int i = 0; i < src.rows; i++) {
    {
      for (int j = 0; j < src.cols; j++) {
        if (HSVchannels[0].at<uchar>(i, j) <= 20 || HSVchannels[0].at<uchar>(i, j) >= 120)  //((normRGBChannels[0].at<float>(i, j) / normRGBChannels[1].at<float>(i, j)) >= 1 && ((HSVchannels[0].at<uchar>(i, j) <= 20 || HSVchannels[0].at<uchar>(i, j) >= 120) && (YCrCbchannels[1].at<uchar>(i, j) >= 128 && YCrCbchannels[1].at<uchar>(i, j) <= 173) && (YCrCbchannels[2].at<uchar>(i, j) >= 77 && YCrCbchannels[2].at<uchar>(i, j) <= 132)))
        {
          dst.at<uchar>(i, j) = 255;
        }
      }
    }
  }

  return dst;
}

std::vector<cv::Mat> segmentation::normalizeRGB(cv::Mat RChannel, cv::Mat GChannel, cv::Mat BChannel) {
  if (RChannel.type() != CV_8UC1 || GChannel.type() != CV_8UC1 || BChannel.type() != CV_8UC1) {
    std::cerr << "R,G,B must be CV_8U" << std::endl;
    return cv::Mat();
  }

  cv::Mat normR = cv::Mat::zeros(RChannel.size(), CV_32F);
  cv::Mat normG = cv::Mat::zeros(GChannel.size(), CV_32F);
  cv::Mat normB = cv::Mat::zeros(BChannel.size(), CV_32F);
  for (int h = 0; h < RChannel.rows; h++) {
    for (int k = 0; k < RChannel.cols; k++) {
      int rgb = RChannel.at<uchar>(h, k) + GChannel.at<uchar>(h, k) + BChannel.at<uchar>(h, k);
      if (rgb == 0) {
        normR.at<float>(h, k) = 0.0;
        normG.at<float>(h, k) = 0.0;
        normB.at<float>(h, k) = 0.0;
      } else {
        normR.at<float>(h, k) = (float)RChannel.at<uchar>(h, k) / (float)rgb;
        normG.at<float>(h, k) = (float)GChannel.at<uchar>(h, k) / (float)rgb;
        normB.at<float>(h, k) = (float)BChannel.at<uchar>(h, k) / (float)rgb;
      }
    }
  }

  std::vector<cv::Mat> normRGBChannels;
  normRGBChannels.push_back(normR);
  normRGBChannels.push_back(normG);
  normRGBChannels.push_back(normB);

  return normRGBChannels;
}

cv::Mat segmentation::getMaskIntersectImage(cv::Mat src, cv::Mat mask) {
  if (mask.size() != src.size()) {
    std::cerr << "src and mask must have the same size" << std::endl;
    return cv::Mat();
  }

  if (mask.type() != CV_8UC1) {
    std::cerr << "mask must be CV_8UC1" << std::endl;
    return cv::Mat();
  }

  cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC3);
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (mask.at<uchar>(i, j) == 255) {
        dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i, j);
      }
    }
  }

  return dst;
}