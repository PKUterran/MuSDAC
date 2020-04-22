import torch
import numpy as np
from functools import reduce
from itertools import chain
from .layers import *
from .config import *
from random import random, sample
from torch.nn import Sigmoid, Sequential, LeakyReLU, Parameter


class SoftmaxWeightedAveraging(Module):
    KEEP_RATE = 0.95
    ETA = -25

    def __init__(self, n_channel: int, moving: bool = False, avg: bool = False):
        super(SoftmaxWeightedAveraging, self).__init__()
        self.theta = np.array([[[1 / n_channel] * n_channel]])
        self.moving = moving
        self.avg = avg

    def forward(self, x: torch.FloatTensor):
        assert self.theta.shape[2] == x.shape[1], 'Wrong input shape: x {}'.format(x.shape)
        return torch.squeeze(torch.matmul(torch.tensor(self.theta, dtype=torch.float32).cuda(), x), dim=1)

    def update_theta(self, losses: list):
        if self.avg:
            return
        new_theta = np.array([[(torch.stack(losses) * self.ETA).softmax(0).cpu().tolist()]])
        if self.moving:
            self.theta = self.KEEP_RATE * self.theta + (1 - self.KEEP_RATE) * new_theta
        else:
            self.theta = new_theta


class WeightedSummation(Module):
    def __init__(self, n_channel: int, requires_grad=True, restriction=F.softmax):
        super(WeightedSummation, self).__init__()
        self.theta = Parameter(torch.full([1, 1, n_channel], 0, dtype=torch.float32))
        if not requires_grad:
            self.theta.requires_grad_(False)
        self.restriction = restriction

    def forward(self, x: torch.FloatTensor):
        assert self.theta.shape[2] == x.shape[1], 'Wrong input shape: x {}'.format(x.shape)
        return torch.squeeze(torch.matmul(self.restriction(self.theta, dim=2), x), dim=1)


class AttentionSummation(Module):
    def __init__(self, n_channel: int, emb_dim: int):
        super(AttentionSummation, self).__init__()
        self.theta = torch.full([1, 1, n_channel], 1 / n_channel, dtype=torch.float32)
        self.w_omega = Parameter(torch.normal(torch.full([emb_dim, emb_dim], 0.), 0.1))
        self.b_omega = Parameter(torch.normal(torch.full([emb_dim], 0.), 0.1))
        self.u_omega = Parameter(torch.normal(torch.full([emb_dim], 0.), 0.1))

    def forward(self, x: torch.FloatTensor):
        assert self.theta.shape[2] == x.shape[1], 'Wrong input shape: x {}'.format(x.shape)
        return torch.squeeze(torch.matmul(self.theta, x), dim=1)

    def calc_theta(self, xs: list):
        inputs = torch.transpose(torch.stack(xs, dim=0), 0, 1)
        inputs = inputs.detach()
        v = torch.tanh(torch.tensordot(inputs, self.w_omega, dims=1) + self.b_omega)
        vu = torch.tensordot(v, self.u_omega, dims=1)
        alphas = F.softmax(vu, dim=1)
        self.theta = torch.mean(alphas, dim=0).unsqueeze(0).unsqueeze(0)


class MuCDAC(Module):
    def __init__(self, fea_dim: int, hid1_dim: int, hid2_dim: int, emb_dim: int, cls_dim: int, n_meta: int, cu='cu'):
        super(MuCDAC, self).__init__()
        self.n_meta = n_meta
        self.adjs = []
        self.gc1s = []
        self.gc2s = []
        for i in range(n_meta):
            gc1 = GraphConvolution(fea_dim, hid1_dim, drop=MuCDACConfig.GCDrop, activation=F.leaky_relu)
            gc2 = GraphConvolution(hid1_dim, hid2_dim, drop=MuCDACConfig.GCDrop, activation=lambda x: x)
            self.gc1s.append(gc1)
            self.gc2s.append(gc2)

        if cu == 'cu':
            comps = self.compose(n_meta)
        elif cu == 'c':
            comps = self.compose(n_meta)[n_meta:]
        elif cu == 'u':
            comps = [(i,) for i in range(n_meta)]
        elif cu == 'f':
            comps = [tuple(range(n_meta))]
        elif cu == 'urf':
            comps = self.urf(n_meta)
        elif cu == 'rdm':
            comps = sample(self.compose(n_meta), 2 * n_meta - 1)
        elif cu == 'alc':
            comps = [tuple(range(n_meta))] * (2 * n_meta - 1)
        elif isinstance(cu, list):
            comps = cu
        else:
            assert False, 'Wrong CU Mode: {}'.format(cu)
        self.comps_num = len(comps)
        self.comp_ca_cls_list = []
        for comp in comps:
            ca = ChannelAggregation(hid2_dim, len(comp), emb_dim)
            cls = Classifier(emb_dim, cls_dim, drop=MuCDACConfig.CLSDrop)
            self.comp_ca_cls_list.append((comp, ca, cls))

        for i, param in enumerate(self.get_inner_parameters()):
            self.register_parameter(str(i), param)

    def forward(self, features: torch.FloatTensor) -> (torch.FloatTensor, list, list):
        adjs = self.adjs
        assert self.n_meta == len(adjs), 'Need {} adjacency matrices, but {} given.'.format(self.n_meta, len(adjs))
        hidden2s = []
        for i in range(self.n_meta):
            hidden1 = self.gc1s[i](features, adjs[i])
            hidden2 = self.gc2s[i](hidden1, adjs[i])
            hidden2s.append(hidden2)

        embeddings = []
        predictions = []
        for comp, ca, cls in self.comp_ca_cls_list:
            embedding = ca(self.select(hidden2s, comp))
            prediction = cls(embedding)
            embeddings.append(embedding)
            predictions.append(prediction)

        return embeddings, predictions
        # if self.prob_mode:
        #     pred = self.prob_cls(torch.cat(predictions, dim=1))
        # else:
        #     if self.attn:
        #         self.weighted.calc_theta(embeddings)
        #     pred = self.weighted(torch.transpose(torch.stack(predictions), 0, 1))

    def set_adjs(self, adjs: list):
        self.adjs = adjs

    def get_inner_parameters(self):
        return chain(
            reduce(lambda x, y: chain(x, y), map(lambda x: x.parameters(), self.gc1s)),
            reduce(lambda x, y: chain(x, y), map(lambda x: x.parameters(), self.gc2s)),
            reduce(lambda x, y: chain(x, y), map(lambda x: x[1].parameters(), self.comp_ca_cls_list)),
            reduce(lambda x, y: chain(x, y), map(lambda x: x[2].parameters(), self.comp_ca_cls_list)))

    @staticmethod
    def compose(num: int) -> list:
        if num == 1:
            return [(0,)]
        elif num == 2:
            return [(0,), (1,), (0, 1)]
        elif num == 3:
            return [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        else:
            assert False, 'Not supporting meta-path num >= {}.'.format(4)

    @staticmethod
    def urf(num: int) -> list:
        if num == 1:
            return [(0,)]
        u = [(i,) for i in range(num)]
        f = [tuple(range(num))]
        r = []
        for i in range(num - 2):
            while True:
                v = tuple([i for i in range(num) if random() < 0.5])
                if 1 < len(v) < num and v not in r:
                    break
            r.append(v)
        return u + r + f

    @staticmethod
    def select(array: list, elements: tuple) -> list:
        ret = []
        for i in elements:
            ret.append(array[i])
        return ret


class Discriminator(Module):
    def __init__(self, n_emb):
        super(Discriminator, self).__init__()
        self.emb_dim = n_emb
        self.dis_layers = 2
        self.dis_hid_dim = n_emb
        self.dis_dropout = 0.1
        self.dis_input_dropout = 0.5

        layers = [Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(LeakyReLU(0.2))
                layers.append(Dropout(self.dis_dropout))
        layers.append(Sigmoid())
        self.layers = Sequential(*layers)

    def forward(self, x):
        # assert x.dim() == 2
        # assert x.size(1) == self.emb_dim
        return self.layers(x).view(-1)
