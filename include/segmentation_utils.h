#ifndef SEGM_UTILS_H
#define SEGM_UTILS_H

#include <opencv2/core.hpp>
#include <string>

namespace segmentation {
cv::Mat getMaskIntersectImage(cv::Mat src, cv::Mat mask);

std::vector<cv::Mat> normalizeRGB(cv::Mat RChannel, cv::Mat GChannel, cv::Mat BChannel);
cv::Mat getLargestConnectedComponents(cv::Mat src, int minArea);

cv::Mat grabCutRect(cv::Mat src, int iter, int padding = 0);
cv::Mat SLICSuperPixel(cv::Mat src, int superpixelNumber, int regionSize, float ruler, int minElementSize, int numberIterations);
cv::Mat skinThreshold(cv::Mat srSc);

}  // namespace segmentation

#endif  // SEGM_UTILS_H