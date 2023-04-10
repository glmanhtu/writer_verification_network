import glob
import logging
import os
import random

import imagesize
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset

from utils.data_utils import chunks, load_triplet_file
from utils.transform import MovingResize

logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
dir_path = os.path.dirname(os.path.realpath(__file__))


class TMDataset(Dataset):

    def __init__(self, dataset_path: str, transforms, train_letters, is_train=False, fold=1, k_fold=3,
                 with_likely=False, supervised_training=False, triplet=False, n_samples_per_tm=8):
        self.dataset_path = dataset_path
        self.is_train = is_train
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

        for letter in letters:
            for tm in letters[letter]:
                letters[letter][tm] = list(chunks(letters[letter][tm], n_samples_per_tm))
                letters[letter][tm] = [x for x in letters[letter][tm] if len(x) > 0]

        self.letters = letters

        root_dir = os.path.dirname(dir_path)
        triplet_def = {
            'α': load_triplet_file(os.path.join(root_dir, 'BT120220128.triplet'), letters['α'].keys(), with_likely),
            'ε': load_triplet_file(os.path.join(root_dir, 'Eps20220408.triplet'), letters['ε'].keys(), with_likely),
            'μ': load_triplet_file(os.path.join(root_dir, 'mtest.triplet'), letters['μ'].keys(), with_likely),
        }
        self.triplet_def = triplet_def
        self.positive_def, self.negative_def = {}, {}
        for letter in triplet_def:
            self.positive_def[letter], self.negative_def[letter] = triplet_def[letter]

        self.data = []
        for letter in letters:
            for tm in letters[letter]:
                if triplet and is_train and len(self.negative_def[letter][tm]) == 0:
                    continue
                for idx, anchor_bucket in enumerate(letters[letter][tm]):
                    self.data.append((letter, tm, idx))
        self.transforms = transforms
        self.supervised_training = supervised_training
        self.triplet_training_mode = triplet

    def __len__(self):
        return len(self.data)

    def get_img_by_id(self, letter, tm):
        img_file = random.choice(random.choice(self.letters[letter][tm]))
        transforms = torchvision.transforms.Compose([
            MovingResize((64, 64), random_move=False),
            torchvision.transforms.Resize(224),
            lambda x: np.array(x)
        ])

        with Image.open(img_file) as img:
            return transforms(img)

    def __getitem__(self, idx):
        (letter, tm, anchor_bucket_idx) = self.data[idx]

        anchor = random.choice(self.letters[letter][tm][anchor_bucket_idx])
        with Image.open(anchor) as img:
            anchor_img = self.transforms(img)

        target_tm = tm
        if self.supervised_training:
            target_tm = random.choice(tuple(self.positive_def[letter][tm]))
            positive_samples = random.choice(self.letters[letter][target_tm])
        else:
            pos_id = random.choice([x for x in range(len(self.letters[letter][target_tm])) if x != anchor_bucket_idx])
            positive_samples = self.letters[letter][target_tm][pos_id]

        with Image.open(random.choice(positive_samples)) as img:
            positive_img = self.transforms(img)

        result = {
            "positive": positive_img,
            "anchor": anchor_img,
            "letter": letter,
            "tm": tm,
            "pos_tm": target_tm
        }

        if self.triplet_training_mode and self.is_train:
            negative_tm = random.choice(tuple(self.negative_def[letter][tm]))
            negative_samples = random.choice(self.letters[letter][negative_tm])
            with Image.open(random.choice(negative_samples)) as img:
                negative_img = self.transforms(img)
            result['negative'] = negative_img
            result['neg_tm'] = negative_tm

        return result


