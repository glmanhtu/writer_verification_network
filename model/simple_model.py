import torch
import torch.nn as nn

from model.distance_model import DistanceModel
import torch.nn.functional as F

criterion = nn.CosineSimilarity(dim=1)


class SimpleModel(DistanceModel):
    """
    Build a SimSiam model.
    """

    def compute_distance(self, source_features, target_features):
        return F.mse_loss(source_features, target_features)

    def __init__(self, base_encoder):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimpleModel, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(pretrained=True)
        self.encoder.fc = nn.Identity()

    def forward(self, batch_data, device):
        positive_images = batch_data['positive'].to(device, non_blocking=True)
        anchor_images = batch_data['anchor'].to(device, non_blocking=True)

        # compute features for one view
        z1 = self.encoder(positive_images) # NxC
        z2 = self.encoder(anchor_images) # NxC

        return torch.tensor(0, device=z1.device), (z1, z2)