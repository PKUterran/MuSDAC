import json
import os
import numpy as np


def reduce_score(score_list: list) -> float:
    return score_list[-1]


seeds = range(15, 20)
results = {}
for seed in seeds:
    if seed == 9:
        continue
    files = os.listdir('{}'.format(seed))
    for file in files:
        # if not file.startswith('result_urf'):
        #     continue
        if not file.endswith('-acm-ab.txt'):
            continue
        if not file.startswith('result_urf-mmd') and not file.startswith('result_rdm-mmd') and \
                not file.startswith('result_alc-mmd'):
            continue
        if file.startswith('result_no_mmd_unique'):
            continue
        ac_src_list = []
        ac_tgt_list = []
        for line in open('{}/{}'.format(seed, file)):
            d = json.loads(line)
            ac_src_list.append(d['ac_src'])
            ac_tgt_list.append(d['ac_tgt'])

        # file = file.split('-')[1]
        try:
            ac_src = reduce_score(ac_src_list)
            ac_tgt = reduce_score(ac_tgt_list)
            results.setdefault(file, ([], []))
            results[file][0].append(ac_src)
            results[file][1].append(ac_tgt)
        except IndexError:
            pass

sorted_results = sorted([(k, (sum(src_scores) / len(src_scores), sum(tgt_scores) / len(tgt_scores)),
                          np.std(tgt_scores, ddof=1))
                         for k, (src_scores, tgt_scores) in results.items()], key=lambda x: x[1][1], reverse=True)
std_map = {}
for k, (src, tgt), std in sorted_results:
    print('{:<30} {:.4f} - {:.4f} : {:.4f}'.format(k, src, tgt, std))
    std_map.setdefault(k.split('-')[1], []).append(std)

# for k, v in std_map.items():
#     print('{:<10}  {:.4f}'.format(k, sum(v) / len(v)))
