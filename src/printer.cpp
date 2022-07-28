#include "printer.h"

#include "iostream"

void Printer::print(const Image& image) {
  std::cout << "Image: " << image.getPath() << std::endl;
  try {
    float accuracy = image.getMasksAccuracy();
    std::cout << "Accuracy of the mask: " << accuracy << std::endl;
  } catch (const std::exception& e) {
  }
  try {
    std::vector<float> IOUs = image.getIOUs();
    if (IOUs.size() != 0) {
      std::cout << "IOUs of the detection: " << std::endl;
      for (int i = 0; i < IOUs.size(); ++i) {
        std::cout << "  Hand " << i << ": " << IOUs[i] << std::endl;
      }
    }
  } catch (const std::exception& e) {
  }
  std::cout << std::endl;
}