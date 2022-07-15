import argparse
import pathlib
from objects.dataset import Dataset

cli_parser = argparse.ArgumentParser(
    description='EgoHands and HandOverFaces parser')

cli_parser.add_argument('--ego', '--egohands-path',
                        action='store', help="Path to the EgoHands dataset folder")
cli_parser.add_argument('--hof', '--hand-over-faces-path',
                        action='store', help="Path to the HandOverFaces folder")
cli_parser.add_argument('-o', '--output',
                        action='store', help="Path to the output folder")
# cli_parser.add_argument('-v', '--verbose', action='store_true')


def get_arguments():
  args = cli_parser.parse_args()

  egohands_path = args.ego
  hand_over_faces_path = args.hof
  output_path = args.output

  return egohands_path, hand_over_faces_path, output_path


egohands_path, hand_over_faces_path, output_path = get_arguments()

dataset = Dataset(output_path)

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

dataset.load_egohands(egohands_path, skip_images=egohands_skip)
dataset.load_hand_over_face(
    hand_over_faces_path, skip_images=hand_over_face_skip)

print(len(dataset.images))
print(len(dataset.masks))

dataset.generate_images(pathlib.Path(output_path, 'images'))
dataset.generate_masks(pathlib.Path(output_path, 'masks'), merge=True)
dataset.generate_boxes(pathlib.Path(output_path, 'boxes'), normalize=True)
dataset.generate_hand(pathlib.Path(output_path, 'hand'))

print(len(dataset.splitted_masks))
print(len(dataset.merged_masks))
print(len(dataset.boxes))
print(len(dataset.normalized_boxes))
