import numpy as np
import torch

from nets.layers import MaximumMeanDiscrepancy


def acc(pred, labels):
    result = (np.argmax(pred.cpu().detach().numpy(), 1) == labels.cpu().numpy())
    return np.average(result)


def calc_conditional_mdd(mmd: MaximumMeanDiscrepancy, src: torch.Tensor, tgt: torch.Tensor,
                         labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    cls = labels.max().item() + 1
    source_centroid = torch.stack([src[labels == i].sum(dim=0) for i in range(cls)]) / src.size()[0]
    pseudo_labels = logits.argmax(dim=-1)
    target_centroid = torch.stack([tgt[pseudo_labels == i].sum(dim=0) for i in range(cls)]) / tgt.size()[0]
    return mmd(source_centroid, target_centroid)
