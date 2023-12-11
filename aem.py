import copy
import logging
import os
import tempfile
import time

import albumentations as A
import cv2
import hydra
import torch
import torchvision
from ml_engine.criterion.losses import NegativeCosineSimilarityLoss, LossCombination
from ml_engine.engine import Trainer
from ml_engine.evaluation.distances import compute_distance_matrix
from ml_engine.evaluation.metrics import AverageMeter, calc_map_prak
from ml_engine.modelling.resnet import ResNetWrapper, ResNet32MixConv
from ml_engine.modelling.simsiam import SimSiamV2CE
from ml_engine.preprocessing.transforms import ACompose, RandomResize, PadCenterCrop
from ml_engine.tracking.mlflow_tracker import MLFlowTracker
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from aem_dataset import AEMLetterDataset, AEMDataLoader, load_triplet_file
from criterion import ClassificationLoss, SubSetSimSiamLoss, SubSetTripletLoss

logger = logging.getLogger(__name__)


class AEMTrainer(Trainer):
    def get_transform(self, mode, data_cfg):
        img_size = data_cfg.img_size
        if mode == 'train':
            return torchvision.transforms.Compose([
                torchvision.transforms.RandomApply([
                    torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                ], p=0.5),
                torchvision.transforms.RandomAffine(5, translate=(0.1, 0.1), fill=255),
                ACompose([
                    A.LongestMaxSize(max_size=img_size),
                    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=15, p=0.5, value=(255, 255, 255),
                                       border_mode=cv2.BORDER_CONSTANT),

                ]),
                RandomResize(img_size, ratio=(0.85, 1.0)),
                torchvision.transforms.RandomApply([
                    torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)),
                ], p=0.5),
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
                pretrained=model_conf.pretrained,
                dim=model_conf.embed_dim,
                pred_dim=model_conf.pred_dim,
                dropout=model_conf.dropout,
                n_classes=model_conf.n_classes)

        elif model_conf.type == 'resnet':
            model = ResNetWrapper(
                backbone=model_conf.arch,
                weights=model_conf.pretrained,
                layers_to_freeze=model_conf.layers_freeze)

        elif model_conf.type == 'mixconv':
            model = ResNet32MixConv(
                img_size=(self._cfg.data.img_size, self._cfg.data.img_size),
                backbone=model_conf.arch,
                out_channels=model_conf.out_channels,
                mix_depth=model_conf.mix_depth,
                out_rows=model_conf.out_rows,
                weights=model_conf.pretrained,
                layers_to_freeze=model_conf.layers_freeze)

        else:
            raise NotImplementedError(f'Network {model_conf.type} is not implemented!')

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    def load_dataset(self, mode, data_conf, transform):
        datasets = []
        for letter in data_conf.letters:
            dataset_path = data_conf.path
            dataset = AEMLetterDataset(dataset_path, transform, letter)
            datasets.append(dataset)
        return datasets

    def get_dataloader(self, mode, datasets, data_conf, repeat):
        if mode in self.data_loader_registers:
            return self.data_loader_registers[mode]

        data_loader = []
        if mode == 'train':
            data_loader = AEMDataLoader(datasets, batch_size=data_conf.batch_size,
                                        m=data_conf.m_per_class,
                                        numb_workers=data_conf.num_workers,
                                        pin_memory=data_conf.pin_memory)
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
        if self.is_simsiam():
            ssl = SubSetSimSiamLoss(n_subsets=len(letters), weight=self._cfg.train.combine_loss_weight)
            cls = ClassificationLoss(n_subsets=len(letters), weight=1 - self._cfg.train.combine_loss_weight)
            return LossCombination([ssl, cls])
        return SubSetTripletLoss(margin=0.15, n_subsets=len(letters))

    def is_simsiam(self):
        return 'ss' in self._cfg.model.type

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
                if self.is_simsiam():
                    embs, _, _ = embs

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
        distance_df = compute_distance_matrix(features, reduction=self._cfg.eval.distance_reduction,
                                              distance_fn=NegativeCosineSimilarityLoss())

        tms = []
        dataset_tms = set(distance_df.columns)
        positive_pairs, negative_pairs = copy.deepcopy(triplet_def)
        for tm in list(positive_pairs.keys()):
            if tm in dataset_tms:
                positive_tms = positive_pairs[tm].intersection(dataset_tms)
                if len(positive_tms) > 1:
                    tms.append(tm)

        categories = sorted(tms)
        distance_df = distance_df.loc[categories, categories]

        distance_matrix = distance_df.to_numpy()
        m_ap, (top_1, pr_a_k5) = calc_map_prak(distance_matrix, distance_df.columns, positive_pairs, negative_pairs)

        m_ap_meter.update(m_ap)
        top1_meter.update(top_1)
        pk5_meter.update(pr_a_k5)

        AverageMeter.reduces(m_ap_meter, top1_meter, pk5_meter)

        if 1 - m_ap_meter.avg < self._min_loss:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, 'distance_matrix.csv')
                distance_df.to_csv(path)
                self._tracker.log_artifact(path, 'best_results')

        return m_ap_meter.avg, top1_meter.avg, pk5_meter.avg, tms

    def validate_one_epoch(self, dataloaders):
        final_map, final_top1, final_pra5 = [], [], []
        for idx, letter in enumerate(self._cfg.data.letters):
            triplet_def = load_triplet_file(self._cfg.data.triplet_files[idx], self._cfg.data.with_likely)

            m_ap, top1, pra5, tms = self.validate_dataloader(dataloaders[idx], triplet_def)
            self.log_metrics({f'{letter}_mAP': m_ap, f'{letter}_top1': top1, f'{letter}_prak5': pra5})
            logger.info(
                f'Letter {letter}:\t'
                f'N TMs: {len(tms)}\t' 
                f'mAP {m_ap:.4f}\t'
                f'top1 {top1:.3f}\t'
                f'pr@k10 {pra5:.3f}\t')

            final_map.append(m_ap)
            final_top1.append(top1)
            final_pra5.append(pra5)

        final_map = sum(final_map) / len(final_map)
        final_top1 = sum(final_top1) / len(final_top1)
        final_pra5 = sum(final_pra5) / len(final_pra5)

        self.log_metrics({'mAP': final_map, 'top1': final_top1, 'prak5': final_pra5})
        logger.info(
            f'Average:'
            f'mAP {final_map:.4f}\t'
            f'top1 {final_top1:.3f}\t'
            f'pr@k5 {final_pra5:.3f}\t')
        return 1 - final_map


@hydra.main(version_base=None, config_path="conf", config_name="config")
def dl_main(cfg: DictConfig):
    tracker = MLFlowTracker(cfg.exp.name, cfg.exp.tracking_uri, tags=cfg.exp.tags)
    trainer = AEMTrainer(cfg, tracker)
    with tracker.start_tracking(run_id=cfg.run.run_id, run_name=cfg.run.name, tags=dict(cfg.run.tags)):
        if cfg.mode == 'eval':
            trainer.validate()
        elif cfg == 'throughput':
            trainer.throughput()
        else:
            trainer.train()


if __name__ == '__main__':
    dl_main()
