# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


criterion = nn.TripletMarginLoss()


class TripletNetwork(nn.Module):
    def __init__(self, base_encoder, dim=512):
        super(TripletNetwork, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        # Modify the average pooling layer to use a smaller kernel size
        self.encoder.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, anchor, positive, negative):
        anc = self.encoder(anchor)
        pos = self.encoder(positive)
        neg = self.encoder(negative)

        loss = criterion(anc, pos, neg)
        return loss, (anc, pos)
