import logging

import cv2
import numpy as np
import torchvision.transforms

from dataset.tm_dataset import TMDataset
from options.train_options import TrainOptions
from utils.data_utils import padding_image
from utils.transform import MovingResize

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')

args = TrainOptions(save_conf=False).parse()
img_size = 224
transforms = torchvision.transforms.Compose([MovingResize((64, 64)),
                                             torchvision.transforms.Resize(int(64 * 1.5)),
                                             torchvision.transforms.RandomCrop(img_size, pad_if_needed=True, fill=(255, 255, 255)),
                                             torchvision.transforms.GaussianBlur(3, sigma=(1, 2)),
                                             lambda x: np.array(x)])

train_dataset = TMDataset(args.tm_dataset_path, transforms, args.letters)

for item in train_dataset:
    image = np.concatenate([item['positive'], item['anchor']], axis=0)
    # image = cv2.bitwise_not(image)
    cv2.imshow('image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    cv2.waitKey(50)

    # cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()
