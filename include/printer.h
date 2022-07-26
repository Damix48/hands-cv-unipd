#ifndef PRINTER_H
#define PRINTER_H

#include "image.h"

class Printer {
 public:
  static void print(const Image& image);
};
#endif  // PRINTER_H