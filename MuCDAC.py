import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import json
import math
import numpy as np
from itertools import chain

from nets.layers import MaximumMeanDiscrepancy
from nets.models import MuCDAC, WeightedSummation, AttentionSummation, MLP
from utils.data_reader import load_data
from utils.tendency import plt_tendency, plt_compare
from utils.classifier import write_files, classify
from utils.process import acc, ConditionalMMD

# parser = argparse.ArgumentParser()
# parser.parse_args()

use_cuda = True
n_meta = 3

adjs_a, features_a, labels_a = [], torch.Tensor(), torch.Tensor()
adjs_b, features_b, labels_b = [], torch.Tensor(), torch.Tensor()
features_a *= 1e4
features_b *= 1e4

fea_dim = 0
hid1_dim = 64
hid2_dim = 32
emb_dim = 16
cls_dim = 0
mmd_ratio_ = 10
cmmd_ratio_ = 1
cod_ratio_ = 0.5
grow = 1

lr = 1e-3
wc = 5e-4

seeds = [15, 16, 17, 18, 19]


# seeds = [19]


def load_dataset(src, tgt, meta=3):
    global adjs_a, features_a, labels_a, adjs_b, features_b, labels_b, fea_dim, cls_dim, n_meta
    n_meta = meta
    adjs_a, features_a, labels_a = load_data('data', src, n_meta)
    adjs_b, features_b, labels_b = load_data('data', tgt, n_meta)
    features_a *= 1e4
    features_b *= 1e4
    fea_dim = features_a.shape[-1]
    cls_dim = labels_a.max().item() + 1
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


def train(epochs=200, kernel_num=5, use_mmd=True, avg=False, conditional=False, cu='cu', attn=True, visualize=False,
          voting='',
          tag='', directory='temp/', print_mode=False) -> str:
    if avg:
        attn = False
    model = MuCDAC(fea_dim, hid1_dim, hid2_dim, emb_dim, cls_dim, n_meta, cu)
    if attn:
        summation = AttentionSummation(model.comps_num, emb_dim)
    else:
        summation = WeightedSummation(model.comps_num, not avg)
    mmd = MaximumMeanDiscrepancy(kernel_num=kernel_num)
    cmmd = ConditionalMMD(mmd)

    parameters = chain(model.parameters(), summation.parameters())
    optimizer = optim.Adam(parameters, lr=lr, weight_decay=wc)
    if use_cuda:
        model.cuda()
        summation.cuda()

    def train_one(epoch_rate: float, print_mode=True) -> (float, float):
        ratio = (2 / (1 + math.exp(-grow * epoch_rate)) - 1)
        mmd_ratio = ratio * mmd_ratio_
        cmmd_ratio = ratio * cmmd_ratio_
        cod_ratio = ratio * cod_ratio_
        # print('mmd_ratio:', mmd_ratio)
        optimizer.zero_grad()
        model.set_adjs(adjs_a)
        embs_a, preds_a = model(features_a)
        model.set_adjs(adjs_b)
        embs_b, preds_b = model(features_b)

        if attn:
            summation.calc_theta([torch.cat([emb_a, emb_b]) for emb_a, emb_b in zip(embs_a, embs_b)])
            if print_mode:
                print(summation.theta.cpu().data)
        else:
            if print_mode:
                print(F.softmax(summation.theta, dim=2).cpu().data)

        if use_mmd:
            losses = []
            for src, tgt, pred_a, pred_b in zip(embs_a, embs_b, preds_a, preds_b):
                ls = F.nll_loss(pred_a, labels_a) + mmd_ratio * mmd(src, tgt)
                if conditional:
                    ls += cmmd_ratio * cmmd.calc_cmmd(src, tgt, labels_a, pred_b)
                losses.append(ls)
        else:
            losses = [F.nll_loss(pred, labels_a) for pred in preds_a]
        if print_mode:
            print('losses:', [l.cpu().item() for l in losses])
        loss = summation(torch.stack(losses).unsqueeze(-1).unsqueeze(0)).sum()
        if voting == 'cod':
            loss += cod_ratio * torch.max(summation.theta)
        elif voting == 'dic':
            loss -= cod_ratio * torch.max(summation.theta)
        loss.backward()
        optimizer.step()

        pred_a = summation(torch.transpose(torch.stack(preds_a), 0, 1))
        pred_b = summation(torch.transpose(torch.stack(preds_b), 0, 1))
        a_src = acc(pred_a, labels_a)
        a_tgt = acc(pred_b, labels_b)
        as_src = [acc(pred, labels_a) for pred in preds_a]
        as_tgt = [acc(pred, labels_b) for pred in preds_b]
        if print_mode:
            print('src:', a_src)
            print('tgt:', a_tgt)
            print(as_src)
            print(as_tgt)

        return a_src, a_tgt

    file = 'result_{}.txt'.format(tag)
    f = open(directory + file, 'w+')
    print('write file: {}'.format(file))
    for e in range(epochs):
        # print('epoch:', e)
        a_s, a_t = train_one(e / epochs, print_mode=e == epochs - 1 or print_mode)
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
    for source, target, meta_tag in [
        ('acm_4_1500_a', 'acm_4_1500_b', 'acm-ab'),
        ('acm_4_1500_b', 'acm_4_1500_a', 'acm-ba'),
        # ('slap_4_2000_a', 'slap_4_2000_b', 'slap-ab'),
        # ('slap_4_2000_b', 'slap_4_2000_a', 'slap-ba'),
        ('am_4_1500_a', 'am_4_1500_b', 'am-ab'),
        ('am_4_1500_b', 'am_4_1500_a', 'am-ba'),
        ('dblp_4_1500_a', 'dblp_4_1500_b', 'dblp-ab'),
        ('dblp_4_1500_b', 'dblp_4_1500_a', 'dblp-ba'),
    ]:
        load_dataset(source, target, meta=6 if meta_tag.startswith('slap') else 3)
        kn = 2 if meta_tag.startswith('am') else 5
        for seed in seeds:
            print('----- for seed {} -----'.format(seed))
            set_seed(seed)
            directory = 'temp/{}/'.format(seed)
            # res_fus = train(tag='fus-am-ba', cu='f', avg=True, directory=directory)
            # res_cod = train(tag='cod-am-ab', voting='cod', directory=directory)
            # res_dic = train(tag='dic-am-ab', voting='dic', directory=directory)
            # res_no_mmd = train(tag='no_mmd-dblp-ab', use_mmd=False, directory=directory)
            # res_mmd = train(tag='mmd-am-ab', directory=directory)

            # cod_ratio_ = 0.1
            # res_cod = train(tag='urf-cod01-' + meta_tag, voting='cod', directory=directory, cu='urf', kernel_num=kn)
            # cod_ratio_ = 1.0
            # res_cod = train(tag='urf-cod1-' + meta_tag, voting='cod', directory=directory, cu='urf', kernel_num=kn)
            # cod_ratio_ = 2.0
            # res_cod = train(tag='urf-cod2-' + meta_tag, voting='cod', directory=directory, cu='urf', kernel_num=kn)
            #
            # mmd_ratio_ = 1
            # res_mmd = train(tag='urf-mmd1-' + meta_tag, directory=directory, cu='urf', kernel_num=kn)
            # mmd_ratio_ = 5
            # res_mmd = train(tag='urf-mmd5-' + meta_tag, directory=directory, cu='urf', kernel_num=kn)
            # mmd_ratio_ = 20
            # res_mmd = train(tag='urf-mmd20-' + meta_tag, directory=directory, cu='urf', kernel_num=kn)

            # res_uff = train(tag='urf-mmd-' + meta_tag, directory=directory, cu='urf', kernel_num=kn)
            res_rdm = train(tag='rdm-mmd-' + meta_tag, directory=directory, cu='rdm', kernel_num=kn)
            res_alc = train(tag='alc-mmd-' + meta_tag, directory=directory, cu='alc', kernel_num=kn)

            # res_avg = train(tag='urf-avg-' + meta_tag, avg=True, directory=directory, cu='urf', kernel_num=kn)
            # res_mmd = train(tag='urf-mmd-' + meta_tag, directory=directory, cu='urf', kernel_num=kn)
            # res_cod = train(tag='urf-cod-' + meta_tag, voting='cod', directory=directory, cu='urf', kernel_num=kn)
            # res_dic = train(tag='urf-dic-' + meta_tag, voting='dic', directory=directory, cu='urf', kernel_num=kn)
            # res_nom = train(tag='urf-nom-' + meta_tag, use_mmd=False, directory=directory, cu='urf', kernel_num=kn)
            # res_cmmd = train(tag='cmmd-acm-ab', conditional=True, directory=directory)
            # res_avg = train(tag='avg-am-ab', avg=True, directory=directory)
            # res_no_mmd_unique = train(tag='no_mmd_unique-acm-ab', use_mmd=False, cu='u', directory=directory)
            # res_unique = train(tag='mmd_unique-dblp-ba', cu='u', directory=directory)
            # res_common = train(tag='mmd_common-dblp-ba', cu='c', directory=directory)
            # plt_compare([res_no_mmd, res_mmd, res_cmmd, res_avg, res_no_mmd_unique, res_unique, res_common],
            #             tag='acm-ba', directory=directory)
