import pathlib
import shutil
import scipy.io
import numpy as np
import cv2
import xml.etree.ElementTree as ET


class Dataset:
  def __init__(self, _path) -> None:
    self.path = pathlib.Path(_path)

    self.images = []
    self.masks = []
    self.boxes = []

    self.splitted_masks = []
    self.merged_masks = []
    self.normalized_boxes = []

  def load_egohands(self, dataset_path, skip_images=[]):
    images_folder = pathlib.Path(dataset_path, '_LABELLED_SAMPLES')
    metadata_path = pathlib.Path(dataset_path, 'metadata.mat')

    skip_indeces = []

    images_paths = list(images_folder.glob('**/*.jpg'))
    images_paths.sort()

    skip_images.sort()

    skip_index = 0
    for i in range(len(images_paths)):
      image_path = images_paths[i]

      if (len(skip_images) != 0 and skip_index != len(skip_images) and str(pathlib.Path(skip_images[skip_index])) in str(image_path)):
        skip_indeces.append(i)
        skip_index += 1
      else:
        self.images.append(image_path)

    metadata = scipy.io.loadmat(metadata_path)

    skip_index = 0
    for frames in metadata.get('video')[0]:
      for frame in frames[6][0]:
        if skip_index not in skip_indeces:
          masks = []

          for i in range(1, 5):
            hand = frame[i]

            if hand.shape != (0, 0):
              hand = hand.astype(np.int32)
              hand = hand.reshape((-1, 1, 2))
              masks.append(hand)

          self.masks.append(masks)

        skip_index += 1

  def load_hand_over_face(self, dataset_path, skip_images=[]):
    images_folder = pathlib.Path(dataset_path, 'images_original_size')
    metadata_path = pathlib.Path(dataset_path, 'annotations')

    images_paths = list(images_folder.glob('**/*.jpg'))
    images_paths.sort()

    for image_path in images_paths:
      if(image_path.stem not in skip_images):
        self.images.append(image_path)

    masks_paths = list(metadata_path.glob('**/*.xml'))
    masks_paths.sort()

    for mask_path in masks_paths:
      if (mask_path.stem not in skip_images):
        xml = ET.parse(str(mask_path))
        root = xml.getroot()

        polygons = root.findall('./object/polygon')

        masks = []

        for mask in polygons:
          hand = []

          for point in mask.findall('./pt'):
            hand.append([int(point[0].text), int(point[1].text)])

          hand = np.array(hand).astype(np.int32)
          hand = hand.reshape((-1, 1, 2))

          masks.append(hand)

        self.masks.append(masks)

  def generate_images(self, output_path):
    for i in range(len(self.images)):
      out_path = output_path.joinpath('img_' + str(i).zfill(5) + '.jpg')
      pathlib.Path(out_path.parents[0]).mkdir(parents=True, exist_ok=True)

      shutil.copyfile(self.images[i], out_path)

      self.images[i] = out_path

    return None

  def generate_masks(self, output_path, merge=False):
    for i in range(len(self.masks)):
      masks = self.masks[i]

      img = cv2.imread(str(self.images[i]))

      width = img.shape[1]
      height = img.shape[0]

      masks_path = []

      if(merge):
        masks_image = np.zeros((height, width, 1), np.uint8)

      for j in range(len(masks)):
        hand = masks[j]

        out_path = output_path.joinpath(
            'img_' + str(i).zfill(5) + '_' + str(j).zfill(2) + '.png')
        pathlib.Path(out_path.parents[0]).mkdir(parents=True, exist_ok=True)

        mask = np.zeros((height, width, 1), np.uint8)
        cv2.fillPoly(mask, [hand], 1, 1)

        cv2.imwrite(str(out_path), mask*255)

        masks_path.append(out_path)

        if(merge):
          cv2.fillPoly(masks_image, [hand], 1, 1)

      self.splitted_masks.append(masks_path)

      if(merge):
        out_path = pathlib.Path(output_path.parents[0], (
            output_path.name + '_merged'), ('img_' + str(i).zfill(5) + '.png'))
        pathlib.Path(out_path.parents[0]).mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_path), masks_image*255)

        self.merged_masks.append(out_path)

  def generate_boxes(self, output_path, normalize=False):
    for i in range(len(self.splitted_masks)):
      boxes = []

      if(normalize):
        normalized_boxes = []

      for mask in self.splitted_masks[i]:
        img = cv2.imread(str(mask), cv2.CV_8U)

        box = cv2.boundingRect(img)

        boxes.append(box)

        if(normalize):
          width = img.shape[1]
          height = img.shape[0]

          x, y, w, h = box
          x = (x + (w / 2)) / width
          y = (y + (h / 2)) / height
          w = w / width
          h = h / height

          normalized_boxes.append([x, y, w, h])

      self.boxes.append(boxes)

      out_path = output_path.joinpath('img_' + str(i).zfill(5) + '.txt')
      pathlib.Path(out_path.parents[0]).mkdir(parents=True, exist_ok=True)

      with open(out_path, 'w') as f:
        content = ''

        for box in self.boxes[i]:
          content = content + ('{}\t{}\t{}\t{}'.format(*box)) + '\n'

        f.write(content)

      if(normalize):
        self.normalized_boxes.append(normalized_boxes)

        out_path = pathlib.Path(
            output_path.parents[0], (output_path.name + '_normalized'), ('img_' + str(i).zfill(5) + '.txt'))
        print(out_path)
        pathlib.Path(out_path.parents[0]).mkdir(parents=True, exist_ok=True)

        with open(out_path, 'w') as f:
          content = ''

          for box in self.normalized_boxes[i]:
            content = content + ('{}\t{}\t{}\t{}'.format(*box)) + '\n'

          f.write(content)
