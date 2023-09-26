import os.path
import os.path

import torch
from torch.utils.data import DataLoader

import wandb
from dataset.tm_dataset import TMDataset
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils.data_utils import letter_ascii
from utils.misc import compute_similarity_matrix
from utils.transform import val_transforms

args = TrainOptions().parse()
dir_path = os.path.dirname(os.path.realpath(__file__))

wandb.init(group=args.group,
           name=args.name,
           project=args.wb_project,
           entity=args.wb_entity,
           resume=args.resume,
           config=args,
           mode=args.wb_mode)


class Trainer:
    def __init__(self):
        device = torch.device('cuda' if args.cuda else 'cpu')

        self._working_dir = os.path.join(args.checkpoints_dir, args.name)
        self._model = ModelsFactory.get_model(args, self._working_dir, is_train=True, device=device,
                                              dropout=args.dropout)
        transforms = val_transforms(args.image_size)
        dataset_val = TMDataset(args.tm_dataset_path, transforms, ['α', 'ε', 'μ'])

        self.data_loader_val = DataLoader(dataset_val, shuffle=False, num_workers=args.n_threads_test,
                                          batch_size=args.batch_size)

        print("Validating sets: {} images".format(len(dataset_val)))
        self._current_step = 0

    def is_trained(self):
        return self._model.existing()

    def set_current_step(self, step):
        self._current_step = step

    def load_pretrained_model(self):
        self._model.load()

    @staticmethod
    def add_features(letter_features, letters, cliplet_ids, features):
        for letter, cliplet, features in zip(letters, cliplet_ids, features):
            feature_cpu = features.cpu()
            letter_features.setdefault(letter, {}).setdefault(cliplet, []).append(feature_cpu)

    def validate(self, n_time_validates=1):
        # set model to eval
        self._model.set_eval()
        val_losses = []
        letter_features = {}
        for i in range(n_time_validates):
            for i_train_batch, batch in enumerate(self.data_loader_val):
                val_loss, (pos_features, anc_features) = self._model.compute_loss(batch)
                val_losses.append(val_loss.item() + 1)  # negative cosine similarity has range [-1, 1]
                self.add_features(letter_features, batch['letter'], batch['positive_id'], pos_features)
                self.add_features(letter_features, batch['letter'], batch['anchor_id'], anc_features)
            print(f'Finished the evaluating {i + 1}/{n_time_validates}')

        for letter in letter_features:
            similar_df = compute_similarity_matrix(letter_features[letter])
            ascii_letter = letter_ascii[letter]
            similar_df.to_csv(os.path.join(self._working_dir, f'similarity_cliplet_{ascii_letter}.csv'),
                              encoding='utf-8')


if __name__ == "__main__":
    trainer = Trainer()
    if trainer.is_trained():
        trainer.set_current_step(wandb.run.step)
        trainer.load_pretrained_model()

    if not trainer.is_trained():
        raise Exception('Please train your model first')

    trainer.load_pretrained_model()
    trainer.validate(20)
