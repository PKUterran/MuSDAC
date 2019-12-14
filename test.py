import torch
import torch.nn.functional as F
from nets.models import MuCDAC
from nets.layers import MaximumMeanDiscrepancy
from utils.process import ConditionalMMD

# src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# tgt = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# labels = torch.tensor([0, 1], dtype=torch.int64)
# logits = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
# print(torch.cat([src, tgt], dim=-1))
# mmd = MaximumMeanDiscrepancy()
# cmmd = ConditionalMMD(mmd)
# print(mmd(src, tgt))
# print(cmmd.calc_cmmd(src, tgt, labels, logits))


from random import random


def uor(num: int) -> list:
    if num == 1:
        return [(0,)]
    u = [(i,) for i in range(num)]
    o = [tuple(range(num))]
    r = []
    for i in range(num - 2):
        while True:
            v = tuple([i for i in range(num) if random() < 0.5])
            if 1 < len(v) < num and v not in r:
                break
        r.append(v)
    return u + r + o


print(uor(4))
