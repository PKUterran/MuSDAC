import json
import os


def reduce_score(score_list: list) -> float:
    return score_list[-1]


seeds = [1, 2, 3, 4, 5]
results = {}
for seed in seeds:
    files = os.listdir('{}'.format(seed))
    for file in files:
        if not file.startswith('result_'):
            continue
        if not file.endswith('-acm-ab.txt'):
            continue
        ac_src_list = []
        ac_tgt_list = []
        for line in open('{}/{}'.format(seed, file)):
            d = json.loads(line)
            ac_src_list.append(d['ac_src'])
            ac_tgt_list.append(d['ac_tgt'])

        ac_src = reduce_score(ac_src_list)
        ac_tgt = reduce_score(ac_tgt_list)
        results.setdefault(file, ([], []))
        results[file][0].append(ac_src)
        results[file][1].append(ac_tgt)

sorted_results = sorted([(k, (sum(src_scores) / len(src_scores), sum(tgt_scores) / len(tgt_scores)))
                         for k, (src_scores, tgt_scores) in results.items()], key=lambda x: x[1][1], reverse=True)
for k, (src, tgt) in sorted_results:
    print('{:<45} {:.4f} - {:.4f}'.format(k, src, tgt))
