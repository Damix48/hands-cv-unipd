import pathlib
import random
import shutil
import tempfile
import cv2
import argparse
from objects.dataset import Dataset


def create_yolo_dataset(dataset: Dataset, output_path, black_white_ratio=0):
  temp_folder = tempfile.TemporaryDirectory()

  dataset.generate_images(pathlib.Path(temp_folder.name, 'images'))
  dataset.generate_masks(pathlib.Path(temp_folder.name, 'masks'))
  dataset.generate_boxes(pathlib.Path(
      temp_folder.name, 'boxes'), normalize=True)

  files = list(zip(dataset.images, dataset.normalized_boxes_paths))

  random.shuffle(files)

  dataset_size = len(files)
  train_size = dataset_size * 0.7

  images_path = pathlib.Path(output_path, 'images')
  labels_path = pathlib.Path(output_path, 'labels')

  for i in range(dataset_size):
    image_path = files[i][0]
    boxes_path = files[i][1]

    if(i < train_size):
      image_out_path = pathlib.Path(images_path, 'train', image_path.name)
      label_out_path = pathlib.Path(labels_path, 'train', boxes_path.name)
    else:
      image_out_path = pathlib.Path(images_path, 'validation', image_path.name)
      label_out_path = pathlib.Path(labels_path, 'validation', boxes_path.name)

    pathlib.Path(image_out_path.parents[0]).mkdir(parents=True, exist_ok=True)
    pathlib.Path(label_out_path.parents[0]).mkdir(parents=True, exist_ok=True)

    shutil.copy(image_path, image_out_path)
    shutil.copy(boxes_path, label_out_path)

    if(random.random() < black_white_ratio):
      image = cv2.imread(str(image_out_path))
      image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      print(image_out_path)

      cv2.imwrite(str(image_out_path), image_gray)

  temp_folder.cleanup()


cli_parser = argparse.ArgumentParser(
    description='Yolo dataset generator')

cli_parser.add_argument('--ego', '--egohands-path',
                        action='store', help="Path to the EgoHands dataset folder")
cli_parser.add_argument('--hof', '--hand-over-faces-path',
                        action='store', help="Path to the HandOverFaces folder")
cli_parser.add_argument('--gtea', '--gtea-gaze-plus-path',
                        action='store', help="Path to the GTEA Gaze+ folder")
cli_parser.add_argument('--bw', '--black-white-ratio',
                        action='store', help="Ratio from 0-1 for black and white images", type=float, default=0)
cli_parser.add_argument('-o', '--output',
                        action='store', help="Path to the output folder")
# cli_parser.add_argument('-v', '--verbose', action='store_true')


def get_arguments():
  args = cli_parser.parse_args()

  egohands_path = args.ego
  hand_over_faces_path = args.hof
  gtea_gaze_plus_path = args.gtea
  black_white_ratio = args.bw
  output_path = args.output

  return egohands_path, hand_over_faces_path, gtea_gaze_plus_path, black_white_ratio, output_path


if __name__ == "__main__":
  egohands_path, hand_over_faces_path, gtea_gaze_plus_path, black_white_ratio, output_path = get_arguments()

  print(egohands_path, hand_over_faces_path,
        gtea_gaze_plus_path, black_white_ratio, output_path)

  dataset = Dataset()

  egohands_skip = [
      "CARDS_LIVINGROOM_B_T/frame_0504",
      "CARDS_LIVINGROOM_B_T/frame_0677",
      "CARDS_OFFICE_B_S/frame_1915",
      "CARDS_OFFICE_B_S/frame_2591",
      "CARDS_OFFICE_H_T/frame_0233",
      "CARDS_OFFICE_H_T/frame_1875",
      "CHESS_COURTYARD_B_T/frame_0150",
      "CHESS_COURTYARD_B_T/frame_0373",
      "CHESS_COURTYARD_B_T/frame_1039",
      "CHESS_COURTYARD_B_T/frame_1110",
      "CHESS_OFFICE_B_S/frame_0383",
      "CHESS_OFFICE_B_S/frame_1277",
      "CHESS_OFFICE_B_S/frame_0944",
      "JENGA_COURTYARD_B_H/frame_0091",
      "JENGA_LIVINGROOM_H_B/frame_0283",
      "PUZZLE_COURTYARD_B_S/frame_0551",
      "PUZZLE_COURTYARD_B_S/frame_0058",
      "PUZZLE_COURTYARD_B_S/frame_0323",
      "PUZZLE_OFFICE_B_H/frame_0763",
      "PUZZLE_OFFICE_B_H/frame_2377"
  ]

  hand_over_face_skip = [
      "1",
      "2",
      "3",
      "34",
      "38",
      "53",
      "71",  # TROPPE MANI
      "77",
      "146",
      "178",
      "245",
      "216",  # GIF
      "221"  # GIF
  ]

  # dataset.load_egohands(egohands_path, skip_images=egohands_skip)
  dataset.load_hand_over_face(
      hand_over_faces_path, skip_images=hand_over_face_skip)
  # dataset.load_gtea_gaze_plus(gtea_gaze_plus_path)

  create_yolo_dataset(dataset, output_path,
                      black_white_ratio=black_white_ratio)
