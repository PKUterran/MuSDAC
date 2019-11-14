import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import json
import math
from itertools import chain

from nets.layers import MaximumMeanDiscrepancy
from nets.models import MuCDAC, WeightedSummation, AttentionSummation
from utils.data_reader import load_data
from utils.tendency import plt_tendency, plt_compare
from utils.classifier import write_files, classify
from utils.process import acc, calc_conditional_mdd

parser = argparse.ArgumentParser()
parser.parse_args()

use_cuda = True
n_meta = 3

adjs_a, features_a, labels_a = load_data('data', 'acm_4_1500_a', n_meta)
adjs_b, features_b, labels_b = load_data('data', 'acm_4_1500_b', n_meta)
features_a *= 1e4
features_b *= 1e4

fea_dim = features_a.shape[-1]
hid1_dim = 64
hid2_dim = 32
emb_dim = 16
cls_dim = labels_a.max().item() + 1
# mmd_ratio = 1
cmmd_ratio = 1e-1

lr = 1e-3
wc = 5e-4

seeds = {1, 2, 3, 4, 5}

if use_cuda:
    features_a = features_a.cuda()
    features_b = features_b.cuda()
    labels_a = labels_a.cuda()
    labels_b = labels_b.cuda()
    for i in range(n_meta):
        adjs_a[i] = adjs_a[i].cuda()
        adjs_b[i] = adjs_b[i].cuda()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def train(epochs=200, use_mmd=True, avg=False, conditional=False, cu='cu', attn=True, visualize=False,
          tag='', directory='temp/') -> str:
    if avg:
        attn = False
    model = MuCDAC(fea_dim, hid1_dim, hid2_dim, emb_dim, cls_dim, n_meta, cu)
    if attn:
        summation = AttentionSummation(model.comps_num, emb_dim)
    else:
        summation = WeightedSummation(model.comps_num, not avg)
    mmd = MaximumMeanDiscrepancy(kernel_num=2)
    optimizer = optim.Adam(chain(model.parameters(), summation.parameters()), lr, weight_decay=wc)
    if use_cuda:
        model.cuda()
        summation.cuda()

    def train_one(epoch_rate: float) -> (float, float):
        mmd_ratio = 2 / (1 + math.exp(-10 * epoch_rate)) - 1
        # print('mmd_ratio:', mmd_ratio)
        optimizer.zero_grad()
        model.set_adjs(adjs_a)
        embs_a, preds_a = model(features_a)
        model.set_adjs(adjs_b)
        embs_b, preds_b = model(features_b)

        if attn:
            summation.calc_theta([torch.cat([emb_a, emb_b]) for emb_a, emb_b in zip(embs_a, embs_b)])
            print(summation.theta.cpu().data)
        else:
            print(F.softmax(summation.theta, dim=2).cpu().data)

        if use_mmd:
            losses = []
            for src, tgt, pred_a, pred_b in zip(embs_a, embs_b, preds_a, preds_b):
                ls = F.nll_loss(pred_a, labels_a) + mmd_ratio * mmd(src, tgt)
                if conditional:
                    ls += cmmd_ratio * calc_conditional_mdd(mmd, src, tgt, labels_a, pred_b)
                losses.append(ls)
        else:
            losses = [F.nll_loss(pred, labels_a) for pred in preds_a]
        print('losses:', [l.cpu().item() for l in losses])
        loss = summation(torch.stack(losses).unsqueeze(-1).unsqueeze(0)).sum()
        loss.backward()
        optimizer.step()

        pred_a = summation(torch.transpose(torch.stack(preds_a), 0, 1))
        pred_b = summation(torch.transpose(torch.stack(preds_b), 0, 1))
        a_src = acc(pred_a, labels_a)
        a_tgt = acc(pred_b, labels_b)
        print('src:', a_src)
        print('tgt:', a_tgt)
        # as_src = [acc(pred, labels_a) for pred in preds_a]
        # as_tgt = [acc(pred, labels_b) for pred in preds_b]
        # print(as_src)
        # print(as_tgt)

        return a_src, a_tgt

    file = 'result_{}.txt'.format(tag)
    f = open(directory + file, 'w+')
    for e in range(epochs):
        print('epoch:', e)
        a_s, a_t = train_one(e / epochs)
        f.write(json.dumps({'epoch': e, 'ac_src': a_s, 'ac_tgt': a_t}) + '\n')

    f.close()
    # plt_tendency(file, tag, directory)
    if visualize:
        model.set_adjs(adjs_a)
        emb_a, _ = model(features_a)
        model.set_adjs(adjs_b)
        emb_b, _ = model(features_b)
        for i, (src, tgt) in enumerate(zip(emb_a, emb_b)):
            write_files(tag + str(i), src.cpu().detach().numpy(), tgt.cpu().detach().numpy(), labels_a.cpu().numpy(),
                        labels_b.cpu().numpy(), directory)
            classify(tag + str(i), directory=directory)
    return file


if __name__ == '__main__':
    for seed in seeds:
        set_seed(seed)
        directory = 'temp/{}/'.format(seed)
        res_no_mmd = train(tag='no_mmd-acm-ab', use_mmd=False, directory=directory)
        res_mmd = train(tag='mmd-acm-ab', directory=directory)
        res_cmmd = train(tag='cmmd-acm-ab', conditional=True, directory=directory)
        res_avg = train(tag='avg-acm-ab', avg=True, directory=directory)
        res_no_mmd_unique = train(tag='no_mmd_unique-acm-ab', use_mmd=False, cu='u', directory=directory)
        res_unique = train(tag='mmd_unique-acm-ab', cu='u', directory=directory)
        res_common = train(tag='mmd_common-acm-ab', cu='c', directory=directory)
        plt_compare([res_no_mmd, res_mmd, res_cmmd, res_avg, res_no_mmd_unique, res_unique, res_common],
                    tag='acm-ab', directory=directory)
