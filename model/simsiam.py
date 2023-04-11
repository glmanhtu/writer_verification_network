# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from model.distance_model import DistanceModel
import torch.nn.functional as F

criterion = nn.CosineSimilarity(dim=1)


class SimSiam(DistanceModel):
    """
    Build a SimSiam model.
    """

    def compute_distance(self, source_features, target_features):
        similarity = F.cosine_similarity(source_features, target_features, dim=1)
        similarity_percentage = (similarity.mean() + 1) / 2
        return 1 - similarity_percentage

    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        # Modify the average pooling layer to use a smaller kernel size
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, batch_data, device):
        positive_images = batch_data['positive'].to(device, non_blocking=True)
        anchor_images = batch_data['anchor'].to(device, non_blocking=True)

        # compute features for one view
        z1 = self.encoder(positive_images) # NxC
        z2 = self.encoder(anchor_images) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        z1 = z1.detach()
        z2 = z2.detach()

        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        return loss, (p1, p2)
