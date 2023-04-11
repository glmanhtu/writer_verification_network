import os.path
import os.path
import time

import torch
from torch.utils.data import DataLoader

import wandb
from dataset.tm_dataset import TMDataset
from model.model_factory import ModelsFactory
from options.train_options import TrainOptions
from utils import wb_utils
from utils.data_utils import load_triplet_file, letter_ascii
from utils.misc import EarlyStop, display_terminal, compute_similarity_matrix, get_metrics, display_terminal_eval, \
    random_query_results
from utils.transform import get_transforms, val_transforms
from utils.wb_utils import create_heatmap


class Trainer:
    def __init__(self, args, fold=0, k_fold=3):
        device = torch.device('cuda' if args.cuda else 'cpu')
        self.args = args

        self._working_dir = os.path.join(args.checkpoints_dir, args.name, f'fold_{fold}')
        os.makedirs(self._working_dir, exist_ok=True)
        self._model = ModelsFactory.get_model(args, self._working_dir, is_train=True, device=device,
                                              dropout=args.dropout)
        transforms = get_transforms(args.image_size)
        is_triplet = args.network == 'triplet'
        dataset_train = TMDataset(args.tm_dataset_path, transforms, args.letters, is_train=True, fold=fold,
                                  k_fold=k_fold, with_likely=args.with_likely, supervised_training=args.supervised,
                                  triplet=is_triplet, n_samples_per_tm=args.n_samples_per_tm)
        self.data_loader_train = DataLoader(dataset_train, shuffle=True, num_workers=args.n_threads_train,
                                            batch_size=args.batch_size, drop_last=True, persistent_workers=True,
                                            pin_memory=True)
        transforms = val_transforms(args.image_size)
        dataset_val = TMDataset(args.tm_dataset_path, transforms, ['α', 'ε', 'μ'], is_train=False, fold=fold,
                                k_fold=k_fold, with_likely=args.with_likely, supervised_training=True,
                                triplet=is_triplet, n_samples_per_tm=999)

        self.data_loader_val = DataLoader(dataset_val, shuffle=False, num_workers=args.n_threads_test,
                                          persistent_workers=True, pin_memory=True, batch_size=args.batch_size)

        self.early_stop = EarlyStop(args.early_stop)
        print("Training sets: {} images".format(len(dataset_train)))
        print("Validating sets: {} images".format(len(dataset_val)))

        self._current_step = 0

    def is_trained(self):
        return self._model.existing()

    def set_current_step(self, step):
        self._current_step = step

    def load_pretrained_model(self):
        self._model.load()

    def train(self):
        best_m_ap = 0.
        for i_epoch in range(1, self.args.nepochs + 1):
            epoch_start_time = time.time()
            # train epoch
            self._train_epoch(i_epoch)
            if self.args.lr_policy == 'step':
                self._model.lr_scheduler.step()

            if not i_epoch % self.args.n_epochs_per_eval == 0:
                continue

            current_m_ap, similarity_matrices, val_dicts = self._validate(i_epoch, self.data_loader_val,
                                                                          n_time_validates=5)

            if current_m_ap > best_m_ap:
                print("Average mAP improved, from {:.4f} to {:.4f}".format(best_m_ap, current_m_ap))
                best_m_ap = current_m_ap
                wandb.run.summary[f'best_model/avg_mAP'] = current_m_ap
                self._model.save()  # save best model
                for letter in similarity_matrices:
                    similar_df = similarity_matrices[letter]
                    ascii_letter = letter_ascii[letter]
                    similar_df.to_csv(os.path.join(self._working_dir, f'similarity_matrix_{ascii_letter}.csv'),
                                      encoding='utf-8')
                    for key in val_dicts[letter]:
                        wandb.run.summary[f'best_model/{key}'] = val_dicts[letter][key]

                    query_results = random_query_results(similar_df, self.data_loader_val.dataset.triplet_def[letter],
                                                         self.data_loader_val.dataset, letter, n_queries=5, top_k=10)
                    wandb.log({f'val/best_model/{ascii_letter}': wb_utils.generate_query_table(query_results, top_k=10)},
                              step=self._current_step)

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self.args.nepochs, time_epoch, time_epoch / 60, time_epoch / 3600))

            if self.early_stop.should_stop(1 - current_m_ap):
                print(f'Early stop at epoch {i_epoch}')
                break

    def _train_epoch(self, i_epoch):
        self._model.set_train()
        losses = []
        for i_train_batch, train_batch in enumerate(self.data_loader_train):
            iter_start_time = time.time()

            train_loss, _ = self._model(train_batch)
            self._model.optimise_params(train_loss)
            losses.append(train_loss.item() + 1)    # negative cosine similarity has range [-1, 1]

            # update epoch info
            self._current_step += 1

            if self._current_step % self.args.save_freq_iter == 0:
                self._model.print_current_lr()
                save_dict = {
                    'train/loss': sum(losses) / len(losses),
                }
                losses.clear()
                wandb.log(save_dict, step=self._current_step)
                display_terminal(iter_start_time, i_epoch, i_train_batch, len(self.data_loader_train), save_dict)

    @staticmethod
    def add_features(letter_features, letters, tm_features, all_features):
        for letter, tm, features in zip(letters, tm_features, all_features.detach()):
            letter_features.setdefault(letter, {}).setdefault(tm, []).append(features)

    def _validate(self, i_epoch, val_loader, mode='val', n_time_validates=1):
        val_start_time = time.time()
        # set model to eval
        self._model.set_eval()
        val_losses = []
        letter_features = {}
        for i in range(n_time_validates):
            for i_train_batch, batch in enumerate(val_loader):
                val_loss, (pos_features, anc_features) = self._model(batch)
                val_losses.append(val_loss.item() + 1)  # negative cosine similarity has range [-1, 1]
                self.add_features(letter_features, batch['letter'], batch['pos_tm'], pos_features)
                self.add_features(letter_features, batch['letter'], batch['tm'], anc_features)
            print(f'Finished the evaluating {i + 1}/{n_time_validates}')

        all_m_ap = []
        similarity_matrices = {}
        val_dicts = {}
        for letter in list(letter_features.keys()):
            ascii_letter = letter_ascii[letter]
            letter_features[letter] = {k: torch.stack(v) for k, v in letter_features[letter].items()}
            similar_df = compute_similarity_matrix(letter_features[letter])
            del letter_features[letter]

            wandb.log({f'val/similarity_matrix/{ascii_letter}': wandb.Image(create_heatmap(similar_df))},
                      step=self._current_step)
            m_ap, top1, pr_a_k10, pr_a_k100 = get_metrics(similar_df, val_loader.dataset.triplet_def[letter])

            val_dict = {
                f'{mode}/{ascii_letter}/loss': sum(val_losses) / len(val_losses),
                f'{mode}/{ascii_letter}/m_ap': m_ap,
                f'{mode}/{ascii_letter}/top_1': top1,
                f'{mode}/{ascii_letter}/pr_a_k10': pr_a_k10,
                f'{mode}/{ascii_letter}/pr_a_k100': pr_a_k100
            }
            wandb.log(val_dict, step=self._current_step)
            display_terminal_eval(val_start_time, i_epoch, val_dict)
            all_m_ap.append(m_ap)
            similarity_matrices[letter] = similar_df
            val_dicts[letter] = val_dict

        return sum(all_m_ap) / len(all_m_ap), similarity_matrices, val_dicts


if __name__ == "__main__":
    train_args = TrainOptions().parse()
    wandb.init(group=train_args.group,
               name=train_args.name,
               project=train_args.wb_project,
               entity=train_args.wb_entity,
               resume=train_args.resume,
               config=train_args,
               settings=wandb.Settings(_disable_stats=True),
               mode=train_args.wb_mode)

    trainer = Trainer(train_args)
    if trainer.is_trained():
        trainer.set_current_step(wandb.run.step)
        trainer.load_pretrained_model()

    if train_args.resume or not trainer.is_trained():
        trainer.train()

    trainer.load_pretrained_model()
