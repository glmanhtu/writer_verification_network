from torch import nn


class DistanceModel(nn.Module):
    def compute_distance(self, source_features, target_features):
        raise NotImplementedError()
