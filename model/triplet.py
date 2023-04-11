# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
import torch
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


criterion = nn.TripletMarginLoss(margin=1)


class TripletNetwork(nn.Module):
    def __init__(self, base_encoder, dim=512, trainable_layers=3):
        super(TripletNetwork, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(zero_init_residual=True, pretrained=True)

        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
        if trainable_layers == 5:
            layers_to_train.append("bn1")
        for name, parameter in self.encoder.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True),  # first layer
                                        nn.Linear(prev_dim, dim))
        # Modify the average pooling layer to use a smaller kernel size
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, batch_data, device):
        positive = batch_data['positive'].to(device, non_blocking=True)
        anchor = batch_data['anchor'].to(device, non_blocking=True)

        anc = self.encoder(anchor)
        pos = self.encoder(positive)

        if not self.training:
            return torch.tensor(0), (anc, pos)

        negative = batch_data['negative'].to(device, non_blocking=True)
        neg = self.encoder(negative)

        loss = criterion(anc, pos, neg)
        return loss, (pos, anc)
