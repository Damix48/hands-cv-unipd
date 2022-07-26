#ifndef SEGM_UTILS_H
#define SEGM_UTILS_H

#include <opencv2/core.hpp>
#include <string>

namespace segmentation {

cv::Mat grabCutRect(cv::Mat src, int iter, int padding = 0);

}  // namespace segmentation

#endif  // SEGM_UTILS_H