#include "yolo_detector.h"

#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

#include "hand.h"
#include "image.h"
#include "normalized_box.h"

YoloDetector::YoloDetector(std::string modelPath, int inputSize_) {
  model = cv::dnn::readNetFromONNX(modelPath);
  inputSize = cv::Size(inputSize_, inputSize_);

  outputLayerNames = model.getUnconnectedOutLayersNames();
}

void YoloDetector::detect(Image &img) {
  cv::Mat blob = img.getImageBlob(inputSize);

  model.setInput(blob);

  std::vector<cv::Mat> results;

  model.forward(results, outputLayerNames);

  cv::Mat output = cv::Mat(results[0].size[1], results[0].size[2], CV_32F, results[0].ptr<float>());

  std::vector<cv::Rect> boxes;
  std::vector<NormalizedBox> normalizedBoxes;
  std::vector<int> indices;
  std::vector<float> scores;

  for (int i = 0; i < output.size[0]; ++i) {
    float x_ = output.at<float>(i, 0);  // x coordinate
    float y_ = output.at<float>(i, 1);  // y coordinate
    float w_ = output.at<float>(i, 2);  // width
    float h_ = output.at<float>(i, 3);  // height
    float score = output.at<float>(i, 4);

    try {
      NormalizedBox box = NormalizedBox::fromYolo(x_, y_, w_, h_, inputSize);

      boxes.push_back(box.toRect(img.size()));
      scores.push_back(score);

      normalizedBoxes.push_back(box);
    } catch (const std::exception &e) {
      // std::cerr << e.what() << '\n';

      NormalizedBox box = NormalizedBox(0, 0, 0, 0);

      boxes.push_back(box.toRect(img.size()));
      scores.push_back(0);

      normalizedBoxes.push_back(box);
    }

    indices.push_back(i);
  }

  cv::dnn::NMSBoxes(boxes, scores, 0.3, 0.4, indices);

  for (int index : indices) {
    Hand hand(normalizedBoxes[index]);
    img.addDetectedHand(hand);
  }
}