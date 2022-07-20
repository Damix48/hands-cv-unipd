#include <iostream>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

cv::Mat RegionGrow(cv::Mat srcImage, cv::Point pt, int ch1Thres, int ch2Thres, int ch3Thres,
                   int ch1LowerBind = 0, int ch1UpperBind = 255, int ch2LowerBind = 0,
                   int ch2UpperBind = 255, int ch3LowerBind = 0, int ch3UpperBind = 255);

int main(int argc, const char** argv) {
  const std::string keys =
      "{help h usage ? |      | print this message   }"
      "{@path          |      | image path           }"
      "{folder         |      | image path           }";

  cv::CommandLineParser parser(argc, argv, keys);

  parser.about("Application name v1.0.0");
  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  std::string path = parser.get<std::string>("@path");
  std::string folder = parser.get<std::string>("folder");

  std::vector<std::string> files;

  cv::glob(folder, files);

  for (int i = 0; i < files.size(); i++) {
    path = files[i];

    cv::Mat image = cv::imread(path);
    cv::Mat cvtImage;
    cv::cvtColor(image, cvtImage, cv::COLOR_BGR2HSV);

    cv::SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 10;
    params.maxThreshold = 200;

    // Filter by Area.
    params.filterByArea = false;
    params.minArea = 60;

    // Filter by Circularity
    params.filterByCircularity = false;
    params.minCircularity = 0.1;

    // Filter by Convexity
    params.filterByConvexity = false;
    params.minConvexity = 0.87;

    // Filter by Inertia
    params.filterByInertia = false;
    params.minInertiaRatio = 0.01;

    // Set up detector with params
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

    // Detect blobs

    std::vector<cv::KeyPoint> keypoints;

    cv::Mat imgGray;
    cv::cvtColor(image, imgGray, cv::COLOR_BGR2GRAY);
    detector->detect(imgGray, keypoints);

    cv::Mat imakey;

    std::cout << keypoints.size() << std::endl;

    cv::drawKeypoints(image, keypoints, imakey, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("blob", imakey);

    cv::Point pt = cv::Point(image.cols / 2, image.rows / 2);
    int ch1Thres = 3;
    int ch2Thres = 3;
    int ch3Thres = 10;

    int ch1LowerBind = 0;
    int ch1UpperBind = 255;
    int ch2LowerBind = 0;
    int ch2UpperBind = 255;
    int ch3LowerBind = 0;
    int ch3UpperBind = 255;

    cv::Mat growImage = RegionGrow(cvtImage, pt, ch1Thres, ch2Thres, ch3Thres, ch1LowerBind, ch1UpperBind, ch2LowerBind,
                                   ch2UpperBind, ch3LowerBind, ch3UpperBind);

    cv::imshow("image", image);
    cv::imshow("test", growImage);
    // cv::waitKey(0);

    cv::Mat mask = cv::Mat(growImage.size(), CV_8U);
    mask.setTo(cv::Scalar(cv::GC_PR_BGD));
    mask.setTo(cv::Scalar(cv::GC_PR_FGD), growImage);

    cv::Mat bgdModel, fgdModel;

    cv::grabCut(image, mask, cv::Rect(), bgdModel, fgdModel, 20, cv::GC_EVAL);

    cv::imshow("maschera", mask * 60);

    cv::waitKey(0);
  }
}

cv::Mat RegionGrow(cv::Mat srcImage, cv::Point pt, int ch1Thres, int ch2Thres, int ch3Thres,
                   int ch1LowerBind, int ch1UpperBind, int ch2LowerBind,
                   int ch2UpperBind, int ch3LowerBind, int ch3UpperBind) {
  cv::Point pToGrowing;                                          // The position of the point to be grown
  int pGrowValue = 0;                                            // Gray value of the point to be grown
  cv::Scalar pSrcValue = 0;                                      // Gray value of growth starting point
  cv::Scalar pCurValue = 0;                                      // The gray value of the current growing point
  cv::Mat growImage = cv::Mat::zeros(srcImage.size(), CV_8UC1);  // Create a blank area and fill it with black
                                                                 // Sequential data of growth direction
  int DIR[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}};
  std::vector<cv::Point> growPtVector;             // Growing point stack
  growPtVector.push_back(pt);                      // Push the growth point into the stack
  growImage.at<uchar>(pt.y, pt.x) = 255;           // Mark the growth point
  pSrcValue = srcImage.at<cv::Vec3b>(pt.y, pt.x);  // Record the gray value of the growing point

  while (!growPtVector.empty())  // grow if the growing stack is not empty

  {
    pt = growPtVector.back();  // Take out a growth point
    growPtVector.pop_back();

    // Grow points in eight directions respectively
    for (int i = 0; i < 9; ++i)

    {
      pToGrowing.x = pt.x + DIR[i][0];
      pToGrowing.y = pt.y + DIR[i][1];
      // Check if it is an edge point
      if (pToGrowing.x < 0 || pToGrowing.y < 0 ||
          pToGrowing.x > (srcImage.cols - 1) || (pToGrowing.y > srcImage.rows - 1))
        continue;

      pGrowValue = growImage.at<uchar>(pToGrowing.y, pToGrowing.x);  // The gray value of the current point to be grown
      pSrcValue = srcImage.at<cv::Vec3b>(pt.y, pt.x);

      if (pGrowValue == 0)  // If the marker has not been grown

      {
        pCurValue = srcImage.at<cv::Vec3b>(pToGrowing.y, pToGrowing.x);
        if (pCurValue[0] <= ch1UpperBind && pCurValue[0] >= ch1LowerBind &&  // The upper and lower bounds of the three channels that limit the growth point
            pCurValue[1] <= ch2UpperBind && pCurValue[1] >= ch2LowerBind &&
            pCurValue[2] <= ch3UpperBind && pCurValue[2] >= ch3LowerBind) {
          if (abs(pSrcValue[0] - pCurValue[0]) < ch1Thres &&
              abs(pSrcValue[1] - pCurValue[1]) < ch2Thres &&
              abs(pSrcValue[2] - pCurValue[2]) < ch3Thres)  // grow within the threshold range
          {
            growImage.at<uchar>(pToGrowing.y, pToGrowing.x) = 255;  // marked as white
            growPtVector.push_back(pToGrowing);                     // Push the next growth point into the stack
          }
        }
      }
    }
  }
  return growImage.clone();
}
