import argparse
import logging

import cv2
import numpy as np
import torch
import albumentations as A
import torchvision.transforms
from ml_engine.preprocessing.transforms import ACompose

from dataset.font_dataset import FontDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

parser = argparse.ArgumentParser('Pajigsaw testing script', add_help=False)
parser.add_argument('--data-path', required=True, type=str, help='path to dataset')
parser.add_argument('--bg-path', required=True, type=str, help='path to dataset')
args = parser.parse_args()


transform = torchvision.transforms.Compose([])
stroke_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomAffine(5, translate=(0.1, 0.1), fill=0),
    ACompose([
        A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=15, p=0.5,
                           border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
        A.CoarseDropout()

    ]),
    torchvision.transforms.RandomApply([
        torchvision.transforms.GaussianBlur((3, 3), (2.0, 4.0)),
    ], p=0.7),
])
train_dataset = FontDataset(args.data_path, args.bg_path, FontDataset.Split.TRAIN,
                            stroke_transform, transform, 'a', image_size=(112, 112), n_items_per_class=5)
for img, label in train_dataset:
    image = np.asarray(img)
    # if label[0] == 1:
    #     image = np.concatenate([first_img, second_img], axis=1)
    # elif label[2] == 1:
    #     image = np.concatenate([second_img, first_img], axis=1)
    # elif label[3] == 1:
    #     image = np.concatenate([second_img, first_img], axis=0)
    # elif label[1] == 1:
    #     image = np.concatenate([first_img, second_img], axis=0)
    #
    # else:
    #     image = np.concatenate([first_img, np.zeros_like(first_img), second_img], axis=0)

    # image = cv2.bitwise_not(image)
    cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    cv2.waitKey(100)

    # cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()
