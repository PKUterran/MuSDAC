import torch
import torch.nn.functional as F
from nets.models import MuCDAC
from nets.layers import MaximumMeanDiscrepancy
from utils.process import calc_conditional_mdd

src = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
tgt = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
labels = torch.tensor([0, 1], dtype=torch.int64)
logits = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
mmd = MaximumMeanDiscrepancy()
print(calc_conditional_mdd(mmd, src, tgt, labels, logits))

