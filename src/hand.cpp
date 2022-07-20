#include "hand.h"

#include "normalized_box.h"

Hand::Hand(NormalizedBox box_) : box(box_) {}

NormalizedBox Hand::getBox() const {
  return box;
}