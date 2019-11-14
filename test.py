import torch
import torch.nn.functional as F
from nets.models import MuCDAC


# x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
# adjs = [torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
#         torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
#         torch.tensor([[0.8, 0.2], [0.2, 0.8]])]
#
# model = MuCDAC(3, 4, 5, 6, 7, 3)
# for param in model.get_parameters():
#     print(param.shape)
# c, es, theta = model(x, adjs)
# print(c)
# print(F.softmax(theta.theta, dim=2))

a = torch.tensor([[[1.0, -2.0], [1.0, -3.0]], [[1.0, -4.0], [1.0, -5.0]]])
print(a.norm(p=2, dim=2))
