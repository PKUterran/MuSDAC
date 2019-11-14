from functools import reduce
from itertools import chain
from .layers import *
from .config import *


class MuCDAC(Module):
    def __init__(self, fea_dim: int, hid1_dim: int, hid2_dim: int, emb_dim: int, cls_dim: int, n_meta: int,
                 avg=False, prob_mode=False, unique=False):
        super(MuCDAC, self).__init__()
        if prob_mode:
            avg = True
        self.n_meta = n_meta
        self.prob_mode = prob_mode
        self.adjs = []
        self.gc1s = []
        self.gc2s = []
        for i in range(n_meta):
            gc1 = GraphConvolution(fea_dim, hid1_dim, drop=MuCDACConfig.GCDrop, activation=F.leaky_relu)
            gc2 = GraphConvolution(hid1_dim, hid2_dim, drop=MuCDACConfig.GCDrop, activation=lambda x: x)
            self.gc1s.append(gc1)
            self.gc2s.append(gc2)

        if not unique:
            comps = self.compose(n_meta)
        else:
            comps = [(i,) for i in range(n_meta)]
        self.comp_ca_cls_list = []
        for comp in comps:
            ca = ChannelAggregation(hid2_dim, len(comp), emb_dim)
            cls = Classifier(emb_dim, cls_dim, drop=MuCDACConfig.CLSDrop)
            self.comp_ca_cls_list.append((comp, ca, cls))

        self.weighted = WeightedSummation(len(comps), not avg)
        for param in self.weighted.parameters():
            self.register_parameter('weighted-{}'.format(param.name), param)
        if prob_mode:
            self.prob_cls = Classifier(len(comps) * cls_dim, cls_dim, drop=MuCDACConfig.CLSDrop)
            for param in self.prob_cls.parameters():
                self.register_parameter('prob_cls-{}'.format(param.name), param)
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

        if self.prob_mode:
            pred = self.prob_cls(torch.cat(predictions, dim=1))
        else:
            pred = self.weighted(torch.transpose(torch.stack(predictions), 0, 1))
        return pred, embeddings, predictions

    def set_adjs(self, adjs: list):
        self.adjs = adjs

    def get_inner_parameters(self):
        return reduce(lambda x, y: chain(x, y), [
            reduce(lambda x, y: chain(x, y), map(lambda x: x.parameters(), self.gc1s)),
            reduce(lambda x, y: chain(x, y), map(lambda x: x.parameters(), self.gc2s)),
            reduce(lambda x, y: chain(x, y), map(lambda x: x[1].parameters(), self.comp_ca_cls_list)),
            reduce(lambda x, y: chain(x, y), map(lambda x: x[2].parameters(), self.comp_ca_cls_list))])

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
    def select(array: list, elements: tuple) -> list:
        ret = []
        for i in elements:
            ret.append(array[i])
        return ret
