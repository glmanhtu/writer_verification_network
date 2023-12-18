import os
import time

import albumentations as A
import cv2
import hydra
import numpy as np
import torch
import torchvision
from ml_engine.criterion.losses import DistanceLoss, LossCombination, NegativeCosineSimilarityLoss, NegativeLoss, \
    BatchDotProduct
from ml_engine.evaluation.distances import compute_distance_matrix_from_embeddings
from ml_engine.evaluation.metrics import AverageMeter, calc_map_prak
from ml_engine.preprocessing.transforms import ACompose
from ml_engine.tracking.mlflow_tracker import MLFlowTracker
from omegaconf import DictConfig

from aem import AEMTrainer, SubSetSimSiamV2Loss
from criterion import SubSetSimSiamLoss, ClassificationLoss, SubSetTripletLoss
from dataset.font_dataset import FontDataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def dl_main(cfg: DictConfig):
    tracker = MLFlowTracker(cfg.exp.name, cfg.exp.tracking_uri, tags=cfg.exp.tags)
    trainer = FontTrainer(cfg, tracker)
    with tracker.start_tracking(run_id=cfg.run.run_id, run_name=cfg.run.name, tags=dict(cfg.run.tags)):
        if cfg.mode == 'eval':
            trainer.validate()
        elif cfg == 'throughput':
            trainer.throughput()
        else:
            trainer.train()

        exp_log_dir = os.path.join(cfg.log_dir, cfg.run.name)
        tracker.log_artifacts(exp_log_dir, 'logs')


class FontTrainer(AEMTrainer):
    def get_transform(self, mode, data_cfg):
        img_size = data_cfg.img_size
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
            ACompose([
                A.CLAHE(p=1)
            ]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        stroke_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomAffine(10, translate=(0.1, 0.1), fill=0),
            ACompose([
                A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=20, p=0.5,
                                   border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
                A.AdvancedBlur(blur_limit=5, p=1),
                A.CoarseDropout(max_holes=20, min_holes=10, max_height=10, max_width=10, p=1)
            ]),
        ])
        return transform, stroke_transform

    def load_dataset(self, mode, data_conf, transform):
        transform, stroke_transform = transform
        datasets = []
        for letter in data_conf.letters:
            dataset_path = data_conf.path
            split = FontDataset.Split.from_string(mode)
            dataset = FontDataset(dataset_path, data_conf.background_path, split, stroke_transform, transform, letter,
                                  data_conf.img_size * 2, data_conf.m_per_class)
            datasets.append(dataset)
        return datasets

    def get_criterion(self):
        letters = self._cfg.data.letters
        if self._cfg.model.type == 'ss2ce':
            ssl = SubSetSimSiamLoss(n_subsets=len(letters), weight=self._cfg.train.combine_loss_weight)
            cls = ClassificationLoss(n_subsets=len(letters), weight=1 - self._cfg.train.combine_loss_weight)
            return DistanceLoss(LossCombination([ssl, cls]), NegativeCosineSimilarityLoss(reduction='none'))
        elif self._cfg.model.type == 'ss2':
            ssl = SubSetSimSiamV2Loss(n_subsets=len(letters))
            return DistanceLoss(ssl, NegativeCosineSimilarityLoss(reduction='none'))
        return DistanceLoss(SubSetTripletLoss(margin=0.3, n_subsets=len(letters)),
                            NegativeLoss(BatchDotProduct(reduction='none')))

    def _validate_dataloader(self, data_loader):
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
        groups = {}
        for idx, label in enumerate(labels.numpy()):
            groups.setdefault(label, []).append(idx)

        positive_pairs = {}
        for idx, label in enumerate(labels.numpy()):
            for item in groups[label]:
                positive_pairs.setdefault(idx, set([])).add(item)

        criterion = self.get_criterion()
        distance_matrix = compute_distance_matrix_from_embeddings(embeddings, criterion.compute_distance)
        m_ap, (top_1, pr_a_k5) = calc_map_prak(distance_matrix, np.arange(len(distance_matrix)), positive_pairs)

        m_ap_meter.update(m_ap)
        top1_meter.update(top_1)
        pk5_meter.update(pr_a_k5)

        AverageMeter.reduces(m_ap_meter, top1_meter, pk5_meter)

        return m_ap_meter.avg, top1_meter.avg, pk5_meter.avg

    def validate_one_epoch(self, dataloaders):
        maps, top1s, pra5s = [], [], []
        for idx, let in enumerate(self._cfg.data.letters):
            m_ap, top1, pra5 = self._validate_dataloader(dataloaders[idx])

            self.logger.info(
                f'Letter {let}:\t'
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

        self.logger.info(f'Average: \t mAP {m_ap:.4f}\t top1 {top1:.3f}\t pr@k5 {pra5:.3f}\t')
        return eval_loss


if __name__ == '__main__':
    dl_main()
