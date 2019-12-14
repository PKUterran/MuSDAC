import numpy as np
import torch

from nets.layers import MaximumMeanDiscrepancy


def acc(pred, labels):
    result = (np.argmax(pred.cpu().detach().numpy(), 1) == labels.cpu().numpy())
    return np.average(result)


class ConditionalMMD:
    MOVING_AVERAGE = 0.7

    def __init__(self, mmd: MaximumMeanDiscrepancy):
        self.mmd = mmd
        self.source_centroids = None
        self.target_centroids = None

    def calc_cmmd(self, src: torch.Tensor, tgt: torch.Tensor, labels: torch.Tensor,
                  logits: torch.Tensor) -> torch.Tensor:
        cls = labels.max().item() + 1
        source_centroids = [src[labels == i].sum(dim=0).unsqueeze(-1) / src.size()[0] for i in range(cls)]
        pseudo_labels = logits.argmax(dim=-1)
        target_centroids = [tgt[pseudo_labels == i].sum(dim=0).unsqueeze(-1) / tgt.size()[0] for i in range(cls)]
        # print(source_centroids, target_centroids)
        self.__moving_average_centroids(source_centroids, target_centroids)
        cmmd_loss = sum([self.mmd(sc, tc) for sc, tc in zip(self.source_centroids, self.target_centroids)]) / cls
        return cmmd_loss

    def __moving_average_centroids(self, cur_source_centroids, cur_target_centroids) -> None:
        if not self.source_centroids:
            self.source_centroids = cur_source_centroids
        for i, sc in enumerate(cur_source_centroids):
            self.source_centroids[i].detach_()
            self.source_centroids[i] = self.MOVING_AVERAGE * self.source_centroids[i] + \
                                       (1 - self.MOVING_AVERAGE) * cur_source_centroids[i]

        if not self.target_centroids:
            self.target_centroids = cur_target_centroids
        for i, tc in enumerate(cur_target_centroids):
            self.target_centroids[i].detach_()
            self.target_centroids[i] = self.MOVING_AVERAGE * self.target_centroids[i] + \
                                       (1 - self.MOVING_AVERAGE) * cur_target_centroids[i]
