import glob
import logging
import os
import random

import imagesize
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from utils.data_utils import chunks
from utils.transform import MovingResize

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')


class TMDataset(Dataset):

    def __init__(self, dataset_path: str, transforms, train_letters, is_train=False, fold=1, k_fold=3):
        self.dataset_path = dataset_path
        assert os.path.isdir(self.dataset_path)
        image_pattern = os.path.join(dataset_path, '**', '*.png')
        files = glob.glob(image_pattern, recursive=True)

        letters = {}
        excluded = {}
        for file in files:
            file_name_components = os.path.basename(file).split('_')
            letter, tm = file_name_components[0], file_name_components[1]
            if letter not in train_letters:
                continue
            width, height = imagesize.get(file)
            if width < 16 or height < 16:
                excluded.setdefault('small_size', []).append(file)
                continue
            if '_ex.' in file:
                if os.path.exists(file.replace('_ex', '')):
                    excluded.setdefault('_ex', []).append(file)
                    continue

            letters.setdefault(letter, {}).setdefault(tm, []).append(file)

        all_tms = set([x for y in letters for x in letters[y].keys()])
        folds = list(chunks(sorted(list(all_tms)), k_fold))
        if is_train:
            del folds[fold]
            tm_keep = set([x for y in folds for x in y])
        else:
            tm_keep = set(folds[fold])

        for letter in list(letters.keys()):
            for tm in list(letters[letter].keys()):
                if tm not in tm_keep:
                    del letters[letter][tm]
                elif len(letters[letter][tm]) < 2:
                    excluded.setdefault('nb_img_per_tm', []).append(letters[letter][tm])
                    del letters[letter][tm]

            if len(letters[letter]) == 0:
                del letters[letter]
        print(f'Image Deleted: {", ".join(f"{k}: {len(v)}" for k, v in excluded.items())}')
        self.letters = letters
        self.data = []
        for letter in letters:
            for tm in letters[letter]:
                for anchor in letters[letter][tm]:
                    self.data.append((letter, tm, anchor))
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def get_img_by_id(self, letter, tm):
        img_file = random.choice(self.letters[letter][tm])
        transforms = torchvision.transforms.Compose([
            MovingResize((64, 64), random_move=False),
            torchvision.transforms.Resize(224),
            lambda x: np.array(x)
        ])

        with Image.open(img_file) as img:
            return transforms(img)

    def __getitem__(self, idx):
        (letter, tm, anchor) = self.data[idx]

        with Image.open(anchor) as img:
            anchor_img = self.transforms(img)

        positive_samples = self.letters[letter][tm].copy()
        positive_samples.remove(anchor)

        with Image.open(random.choice(positive_samples)) as img:
            positive_img = self.transforms(img)

        return {
            "positive": positive_img,
            "anchor": anchor_img,
            "letter": letter,
            "tm": tm
        }
