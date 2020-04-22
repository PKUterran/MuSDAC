import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Dropout, Conv1d


class GraphConvolution(Module):
    def __init__(self, in_dim: int, out_dim: int, drop: float = 0.0, bias: bool = False, activation=F.relu):
        super(GraphConvolution, self).__init__()
        self.dropout = Dropout(drop)
        self.linear = Linear(in_dim, out_dim, bias)
        self.activation = activation

    def forward(self, hidden: torch.FloatTensor, adj: torch.FloatTensor) -> torch.FloatTensor:
        assert hidden.shape[0] == adj.shape[0] and adj.shape[0] == adj.shape[1], \
            'Wrong input shape: hidden {}, adj {}'.format(hidden.shape, adj.shape)
        return self.activation(self.linear(self.dropout(torch.matmul(adj, hidden))))


class ChannelAggregation(Module):
    def __init__(self, in_dim: int, n_channel: int, out_dim: int, activation=F.tanh):
        super(ChannelAggregation, self).__init__()
        self.conv = Conv1d(in_dim, out_dim, n_channel, bias=False)
        self.activation = activation

    def forward(self, embeddings: list) -> torch.FloatTensor:
        return self.activation(torch.squeeze(self.conv(torch.stack(embeddings, dim=2)), dim=2))


class Classifier(Module):
    def __init__(self, fea_dim: int, cls_dim: int, drop: float = 0.0, bias=False, activation=F.softmax):
        super(Classifier, self).__init__()
        self.dropout = Dropout(drop)
        self.linear = Linear(fea_dim, cls_dim, bias)
        self.activation = activation

    def forward(self, features):
        return self.activation(self.linear(self.dropout(features)), dim=1)
        # return self.activation(self.linear2(self.linear(self.dropout(features))), dim=1)


class MLP(Module):
    def __init__(self, fea_dim: int, hid_dim: int, cls_dim: int, drop: float = 0.0, bias=False, activation=F.softmax):
        super(MLP, self).__init__()
        self.dropout = Dropout(drop)
        self.linear = Linear(fea_dim, hid_dim, bias)
        self.linear2 = Linear(hid_dim, cls_dim)
        self.activation = activation

    def forward(self, features):
        return self.activation(self.linear2(self.linear(self.dropout(features))), dim=1)


class MaximumMeanDiscrepancy(Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MaximumMeanDiscrepancy, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return

    @staticmethod
    def guassian_kernel(source, target, kernel_mul, kernel_num, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = (total0 - total1).norm(p=2, dim=2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        # loss = torch.mean(XX + YY - XY - YX)
        return loss


if __name__ == '__main__':
    ca1 = ChannelAggregation(3, 3, 3)
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = torch.tensor([[1.0, 3.0, 4.0], [4.0, 5.0, 18.0]])
    c = torch.tensor([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])
    print(ca1([a, b, c]))
    print(torch.stack([a, b, c]))
