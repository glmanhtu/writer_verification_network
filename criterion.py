import torch
from ml_engine.criterion.simsiam import BatchWiseSimSiamLoss
from ml_engine.criterion.triplet import BatchWiseTripletLoss


class SubSetSimSiamLoss(BatchWiseSimSiamLoss):
    def __init__(self, n_subsets=3, weight=1.):
        super().__init__()
        self.n_subsets = n_subsets
        self.weight = weight

    def forward(self, embeddings, targets):
        ps, zs, _ = embeddings
        mini_batch = len(targets) // self.n_subsets
        ps = torch.split(ps, [mini_batch] * self.n_subsets, dim=0)
        zs = torch.split(zs, [mini_batch] * self.n_subsets, dim=0)
        targets = torch.split(targets, [mini_batch] * self.n_subsets, dim=0)

        losses = []
        for p, z, target in zip(ps, zs, targets):
            losses.append(self.forward_impl(p, z, target))

        return self.weight * sum(losses) / len(losses)


class SubSetTripletLoss(BatchWiseTripletLoss):
    def __init__(self, margin=0.1, n_subsets=3, weight=1.):
        super().__init__(margin)
        self.margin = margin
        self.n_subsets = n_subsets
        self.weight = weight

    def forward(self, emb, target):
        mini_batch = len(emb) // self.n_subsets
        embeddings = torch.split(emb, [mini_batch] * self.n_subsets, dim=0)
        targets = torch.split(target, [mini_batch] * self.n_subsets, dim=0)
        losses = []
        for sub_emb, sub_target in zip(embeddings, targets):
            losses.append(self.forward_impl(sub_emb, sub_target, sub_emb, sub_target))

        return self.weight * sum(losses) / len(losses)


class ClassificationLoss(torch.nn.Module):
    def __init__(self, n_subsets=3, weight=1.):
        super().__init__()
        self.n_subsets = n_subsets
        self.criterion = torch.nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, embeddings, targets):
        if self.n_subsets == 1:
            return 0.

        _, _, cls = embeddings
        mini_batch = len(targets) // self.n_subsets

        labels = []
        for i in range(self.n_subsets):
            labels.append(torch.tensor([i] * mini_batch, device=cls.device, dtype=torch.int64))

        labels = torch.cat(labels, dim=0)
        return self.criterion(cls, labels) * self.weight
