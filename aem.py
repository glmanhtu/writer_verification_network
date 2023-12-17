import copy
import os
import tempfile
import time
from collections import OrderedDict

import albumentations as A
import cv2
import hydra
import pandas as pd
import torch
import torchvision
from ml_engine.criterion.losses import NegativeCosineSimilarityLoss, LossCombination, DistanceLoss, BatchDotProduct, \
    NegativeLoss
from ml_engine.engine import Trainer
from ml_engine.evaluation.distances import compute_distance_matrix
from ml_engine.evaluation.metrics import AverageMeter, calc_map_prak
from ml_engine.modelling.resnet import ResNetWrapper, ResNet32MixConv
from ml_engine.modelling.simsiam import SimSiamV2CE, SimSiamV2
from ml_engine.preprocessing.transforms import ACompose, RandomResize, PadCenterCrop
from ml_engine.tracking.mlflow_tracker import MLFlowTracker
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dataset.aem_dataset import AEMLetterDataset, AEMDataLoader, load_triplet_file
from criterion import ClassificationLoss, SubSetSimSiamLoss, SubSetTripletLoss


@hydra.main(version_base=None, config_path="conf", config_name="config")
def dl_main(cfg: DictConfig):
    tracker = MLFlowTracker(cfg.exp.name, cfg.exp.tracking_uri, tags=cfg.exp.tags)
    with tracker.start_tracking(run_id=cfg.run.run_id, run_name=cfg.run.name, tags=dict(cfg.run.tags)):
        trainer = AEMTrainer(cfg, tracker)
        if cfg.mode == 'eval':
            trainer.validate()
        elif cfg == 'throughput':
            trainer.throughput()
        else:
            trainer.train()

        exp_log_dir = os.path.join(cfg.log_dir, cfg.run.name)
        tracker.log_artifacts(exp_log_dir, 'logs')


class AEMTrainer(Trainer):
    def get_transform(self, mode, data_cfg):
        img_size = data_cfg.img_size
        if mode == 'train':
            return torchvision.transforms.Compose([
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                ], p=0.5),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)),
                ], p=0.5),
                torchvision.transforms.RandomAffine(5, translate=(0.1, 0.1), fill=255),
                ACompose([
                    A.LongestMaxSize(max_size=img_size),
                    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=15, p=0.5, value=(255, 255, 255),
                                       border_mode=cv2.BORDER_CONSTANT),
                ]),
                RandomResize(img_size, ratio=(0.85, 1.0)),
                torchvision.transforms.RandomCrop(img_size, pad_if_needed=True, fill=255),
                torchvision.transforms.RandomGrayscale(p=0.3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            return torchvision.transforms.Compose([
                ACompose([
                    A.LongestMaxSize(max_size=img_size),
                ]),
                PadCenterCrop(img_size, pad_if_needed=True, fill=255),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def build_model(self, model_conf):
        if model_conf.type == 'ss2ce':
            model = SimSiamV2CE(
                arch=model_conf.arch,
                pretrained=model_conf.weights,
                dim=model_conf.embed_dim,
                pred_dim=model_conf.pred_dim,
                dropout=model_conf.dropout,
                n_classes=model_conf.n_classes)

        elif model_conf.type == 'ss2':
            model = SimSiamV2(
                arch=model_conf.arch,
                pretrained=model_conf.weights,
                dim=model_conf.embed_dim,
                pred_dim=model_conf.pred_dim,
                dropout=model_conf.dropout)

        elif model_conf.type == 'resnet':
            model = ResNetWrapper(
                backbone=model_conf.arch,
                weights=model_conf.weights,
                layers_to_freeze=model_conf.layers_freeze)

        elif model_conf.type == 'mixconv':
            model = ResNet32MixConv(
                img_size=(self._cfg.data.img_size, self._cfg.data.img_size),
                backbone=model_conf.arch,
                out_channels=model_conf.out_channels,
                mix_depth=model_conf.mix_depth,
                out_rows=model_conf.out_rows,
                weights=model_conf.weights,
                layers_to_freeze=model_conf.layers_freeze)

        else:
            raise NotImplementedError(f'Network {model_conf.type} is not implemented!')

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def prepare_pretrained_model(self, model_type, model_state_dict, pretrained_state_dict):
        if self._cfg.model.type == 'ss2ce' or self._cfg.model.type == 'ss2':
            final_state_dict = []
            imported_count, not_imported_count = 0, 0
            for k, v in model_state_dict.items():
                pretrained_key = k.replace('encoder.', 'model.model.')
                if pretrained_key in pretrained_state_dict:
                    final_state_dict.append((k, pretrained_state_dict[pretrained_key]))
                    imported_count += 1
                else:
                    final_state_dict.append((k, v))
                    not_imported_count += 1
            final_state_dict = OrderedDict(final_state_dict)
            self.logger.info(f"State dict imported: {imported_count}/{not_imported_count + imported_count}")
            return final_state_dict
        return pretrained_state_dict

    def load_dataset(self, mode, data_conf, transform):
        datasets = []
        for letter in data_conf.letters:
            dataset_path = data_conf.path
            if data_conf.val_path and mode == 'validation':
                dataset_path = data_conf.val_path
            dataset = AEMLetterDataset(dataset_path, transform, letter, data_conf.min_size_limit)
            datasets.append(dataset)
        return datasets

    def get_dataloader(self, mode, datasets, data_conf):
        if mode in self.data_loader_registers:
            return self.data_loader_registers[mode]

        data_loader = []
        if mode == 'train':
            data_loader = AEMDataLoader(datasets, batch_size=data_conf.batch_size,
                                        m=data_conf.m_per_class,
                                        numb_workers=data_conf.num_workers,
                                        pin_memory=data_conf.pin_memory,
                                        repeat=data_conf.train_repeat,
                                        repeat_same_class=data_conf.repeat_same_class)
        else:
            for idx, letter in enumerate(data_conf.letters):
                sub_data_loader = DataLoader(datasets[idx], batch_size=data_conf.batch_size,
                                             num_workers=data_conf.num_workers,
                                             pin_memory=data_conf.pin_memory, drop_last=False, shuffle=False)
                data_loader.append(sub_data_loader)
        self.data_loader_registers[mode] = data_loader
        return data_loader

    def get_criterion(self):
        letters = self._cfg.data.letters
        if self._cfg.model.type == 'ss2ce':
            ssl = SubSetSimSiamLoss(n_subsets=len(letters), weight=self._cfg.train.combine_loss_weight)
            cls = ClassificationLoss(n_subsets=len(letters), weight=1 - self._cfg.train.combine_loss_weight)
            return DistanceLoss(LossCombination([ssl, cls]), NegativeCosineSimilarityLoss())
        elif self._cfg.model.type == 'ss2':
            ssl = SubSetSimSiamLoss(n_subsets=len(letters))
            return DistanceLoss(ssl, NegativeCosineSimilarityLoss())

        return DistanceLoss(SubSetTripletLoss(margin=0.15, n_subsets=len(letters)), NegativeLoss(BatchDotProduct()))

    def validate_dataloader(self, data_loader, triplet_def):
        batch_time, m_ap_meter = AverageMeter(), AverageMeter()
        top1_meter, pk5_meter = AverageMeter(), AverageMeter()

        end = time.time()
        embeddings, labels = [], []
        for idx, (images, targets) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=self._cfg.amp_enable):
                embs = self._model(images)
                if self._cfg.model.type == 'ss2ce':
                    embs, _, _ = embs
                elif self._cfg.model.type == 'ss2':
                    embs, _ = embs

            embeddings.append(embs)
            labels.append(targets)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        embeddings = torch.cat(embeddings)
        labels = torch.cat(labels)

        # embeddings = F.normalize(embeddings, p=2, dim=1)
        features = {}
        for feature, target in zip(embeddings, labels.numpy()):
            tm = data_loader.dataset.labels[target]
            features.setdefault(tm, []).append(feature)

        features = {k: torch.stack(v).cuda() for k, v in features.items()}
        criterion = self.get_criterion()
        distance_df = compute_distance_matrix(features, reduction=self._cfg.eval.distance_reduction,
                                              distance_fn=criterion.compute_distance)

        tms = []
        dataset_tms = set(distance_df.columns)
        positive_pairs, negative_pairs = copy.deepcopy(triplet_def)
        for tm in list(positive_pairs.keys()):
            if tm in dataset_tms:
                positive_tms = positive_pairs[tm].intersection(dataset_tms)
                if len(positive_tms) > 1:
                    tms.append(tm)

        categories = sorted(tms)
        distance_eval = distance_df.loc[categories, categories]

        distance_matrix = distance_eval.to_numpy()
        m_ap, (top_1, pr_a_k5) = calc_map_prak(distance_matrix, distance_eval.columns, positive_pairs, negative_pairs)

        m_ap_meter.update(m_ap)
        top1_meter.update(top_1)
        pk5_meter.update(pr_a_k5)

        AverageMeter.reduces(m_ap_meter, top1_meter, pk5_meter)

        return m_ap_meter.avg, top1_meter.avg, pk5_meter.avg, distance_df

    def validate_one_epoch(self, dataloaders):
        maps, top1s, pra5s = [], [], []
        distance_dfs = []
        for idx, let in enumerate(self._cfg.data.letters):
            triplet_def = load_triplet_file(self._cfg.data.triplet_files[idx], self._cfg.data.with_likely)

            m_ap, top1, pra5, distance_df = self.validate_dataloader(dataloaders[idx], triplet_def)
            distance_dfs.append(distance_df)

            self.logger.info(
                f'Letter {let}:\t'
                f'N TMs: {len(distance_df.columns)}\t' 
                f'mAP {m_ap:.4f}\t'
                f'top1 {top1:.3f}\t'
                f'pr@k10 {pra5:.3f}\t')

            maps.append(m_ap)
            top1s.append(top1)
            pra5s.append(pra5)

        m_ap = sum(maps) / len(maps)
        top1 = sum(top1s) / len(top1s)
        pra5 = sum(pra5s) / len(pra5s)

        eval_loss = 1 - m_ap

        self.log_metrics({'mAP': m_ap, 'top1': top1, 'prak5': pra5})
        if eval_loss < self._min_loss:
            self.log_metrics({'best_mAP': m_ap, 'best_top1': top1, 'best_prak5': pra5})
            for idx, let in enumerate(self._cfg.data.letters):
                self.log_metrics({f'{let}_mAP': maps[idx], f'{let}_top1': top1s[idx], f'{let}_prak5': pra5s[idx]})
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, f'distance_matrix_{let}.csv')
                    distance_dfs[idx].to_csv(path)
                    self._tracker.log_artifact(path, 'best_results')

            average_df = pd.concat(distance_dfs).groupby(level=0).mean()
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, f'distance_matrix_avg.csv')
                average_df.to_csv(path)
                self._tracker.log_artifact(path, 'best_results')

        self.logger.info(f'Average: \t mAP {m_ap:.4f}\t top1 {top1:.3f}\t pr@k5 {pra5:.3f}\t')
        return eval_loss


if __name__ == '__main__':
    dl_main()
