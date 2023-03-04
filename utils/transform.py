import random

import numpy as np
import torchvision.transforms
import torchvision.transforms
from PIL import ImageOps, Image
from torchvision.transforms import transforms

from utils.data_utils import resize_image, padding_image


def get_transforms(img_size):
    return torchvision.transforms.Compose([
        MovingResize((64, 64), random_move=True),
        torchvision.transforms.Resize(int(64 * 1.5)),
        torchvision.transforms.RandomCrop(img_size, pad_if_needed=True, fill=(255, 255, 255)),
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ], p=0.3),
        torchvision.transforms.RandomApply([
            torchvision.transforms.GaussianBlur(3, sigma=(1, 2)),
        ], p=0.5),
        # torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomGrayscale(p=0.3),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def val_transforms(img_size):
    return torchvision.transforms.Compose([
        MovingResize((64, 64), random_move=False),
        torchvision.transforms.Resize(int(64 * 1.5)),
        lambda x: np.array(x),
        lambda x: padding_image(x, (img_size, img_size), color=(255, 255, 255)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class UnNormalize(torchvision.transforms.Normalize):
    def __init__(self,mean,std,*args,**kwargs):
        new_mean = [-m/s for m,s in zip(mean,std)]
        new_std = [1/s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def reverse_transform():
    return torchvision.transforms.Compose([
        UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.ToPILImage(),
        lambda image: ImageOps.invert(image)
    ])


class MovingResize:
    def __init__(self, img_size, random_move=True):
        self.img_width, self.img_height = img_size
        self.random_move = random_move

    def __call__(self, img):
        mov = 0.5 if not self.random_move else random.randint(0, 100) / 100.
        width, height = img.size
        ratio_w = self.img_width / width
        ratio_h = self.img_height / height
        scale = min(ratio_h, ratio_w)
        image = resize_image(img, scale).convert('RGB')
        width, height = image.size
        # Find the dominant color
        # dominant_color = bincount_app(np.asarray(img.convert("RGB")))
        # Add image to the background
        result = Image.new(mode="RGB", size=(self.img_width, self.img_height), color=(255, 255, 255))
        result.paste(image, box=(int(mov * (self.img_width - width)), int(mov * (self.img_height - height))))
        return result
