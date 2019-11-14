import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import json

from nets.layers import MaximumMeanDiscrepancy
from nets.models import MuCDAC
from utils.data_reader import load_data
from utils.tendency import plt_tendency, plt_compare
from utils.classifier import write_files, classify

parser = argparse.ArgumentParser()
parser.parse_args()

use_cuda = True
n_meta = 3

adjs_a, features_a, labels_a = load_data('data', 'am_4_1500_b', n_meta)
adjs_b, features_b, labels_b = load_data('data', 'am_4_1500_a', n_meta)
features_a *= 1e4
features_b *= 1e4

fea_dim = features_a.shape[-1]
hid1_dim = 64
hid2_dim = 32
emb_dim = 16
cls_dim = labels_a.max().item() + 1
mmd_ratio = 1e-1

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


def train(epochs=1000, use_mmd=True, avg=False, prob_mode=False, unique=False, visualize=False,
          tag='', directory='temp/') -> str:
    model = MuCDAC(fea_dim, hid1_dim, hid2_dim, emb_dim, cls_dim, n_meta, avg, prob_mode, unique)
    mmd = MaximumMeanDiscrepancy(kernel_num=2)
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=wc)
    if use_cuda:
        model.cuda()

    def train_one() -> (float, float):
        optimizer.zero_grad()
        model.set_adjs(adjs_a)
        pred_a, emb_a, preds_a = model(features_a)
        model.set_adjs(adjs_b)
        pred_b, emb_b, preds_b = model(features_b)
        print(F.softmax(model.weighted.theta, dim=2).cpu().data)
        # pred_loss = model.weighted(
        #         torch.stack([F.nll_loss(pred, labels_a) for pred in preds_a]).unsqueeze(-1).unsqueeze(0)).sum()
        pred_loss = F.nll_loss(pred_a, labels_a)
        if use_mmd:
            # losses = [F.nll_loss(pred, labels_a) + mmd_ratio * mmd(src, tgt)
            #           for pred, src, tgt in zip(preds_a, emb_a, emb_b)]
            mmd_loss = model.weighted(
                torch.stack([mmd(src, tgt) for src, tgt in zip(emb_a, emb_b)]).unsqueeze(-1).unsqueeze(0)).sum()
            print('pred_loss:', pred_loss.cpu().item())
            print('mmd_loss:', mmd_loss.cpu().item())
            loss = pred_loss + mmd_ratio * mmd_loss
        else:
            # losses = [F.nll_loss(pred, labels_a) for pred in preds_a]
            print('pred_loss:', pred_loss.cpu().item())
            loss = pred_loss
        # print('losses:', [l.cpu().item() for l in losses])
        # loss = model.weighted(torch.stack(losses).unsqueeze(-1).unsqueeze(0)).sum()
        loss.backward()
        optimizer.step()

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
        a_s, a_t = train_one()
        f.write(json.dumps({'epoch': e, 'ac_src': a_s, 'ac_tgt': a_t}) + '\n')

    f.close()
    plt_tendency(file, tag, directory)
    if visualize:
        model.set_adjs(adjs_a)
        _, emb_a, _ = model(features_a)
        model.set_adjs(adjs_b)
        _, emb_b, _ = model(features_b)
        for i, (src, tgt) in enumerate(zip(emb_a, emb_b)):
            write_files(tag + str(i), src.cpu().detach().numpy(), tgt.cpu().detach().numpy(), labels_a.cpu().numpy(),
                        labels_b.cpu().numpy(), directory)
            classify(tag + str(i), directory=directory)
    return file


def acc(pred, labels):
    result = (np.argmax(pred.cpu().detach().numpy(), 1) == labels.cpu().numpy())
    return np.average(result)


if __name__ == '__main__':
    for seed in seeds:
        set_seed(seed)
        directory = 'temp/{}/'.format(seed)
        res_mmd = train(tag='mmd-am-ba', directory=directory, visualize=True)
        break
        res_no_mmd = train(tag='no_mmd-dblp-ba', use_mmd=False, directory=directory)
        res_avg = train(tag='avg-dblp-ba', avg=True, directory=directory)
        # res_mmd_prob = train(tag='mmd_prob-acm-ba', prob_mode=True, directory=directory)
        res_unique = train(tag='mmd_unique-dblp-ba', unique=True, directory=directory)
        res_no_mmd_unique = train(tag='no_mmd_unique-dblp-ba', use_mmd=False, unique=True, directory=directory)
        plt_compare([res_mmd, res_no_mmd, res_avg, res_unique, res_no_mmd_unique],
                    tag='dblp-ba', directory=directory)
