from matplotlib import pyplot as plt
import json

COLORS = ['red', 'orange', 'yellow', 'green', 'azure', 'blue', 'purple']


def plt_tendency(res_file='result.txt', tag='', directory='temp/'):
    fig = plt.figure(figsize=(8, 8))
    results = []
    for line in open(directory + res_file):
        results.append(json.loads(line))
    result = {
        'epoch': [i['epoch'] for i in results],
        'ac_src': [i['ac_src'] for i in results],
        'ac_tgt': [i['ac_tgt'] for i in results]
    }
    plt.plot(result['epoch'], result['ac_src'], c='blue')
    plt.plot(result['epoch'], result['ac_tgt'], c='red')
    plt.savefig(directory + 'tendency_{}.png'.format(tag))
    plt.close(0)


def plt_compare(res_files: list, tag='', directory='temp/'):
    fig = plt.figure(figsize=(8, 8))
    results_ = {}
    for res_file in res_files:
        results_[res_file] = []
        for line in open(directory + res_file):
            results_[res_file].append(json.loads(line))
    for i, (res_file, results) in enumerate(results_.items()):
        result = {
            'epoch': [i['epoch'] for i in results],
            'ac_src': [i['ac_src'] for i in results],
            'ac_tgt': [i['ac_tgt'] for i in results]
        }
        plt.plot(result['epoch'], result['ac_src'], c=COLORS[i % len(COLORS)], linestyle='--')
        plt.plot(result['epoch'], result['ac_tgt'], c=COLORS[i % len(COLORS)])

    plt.savefig(directory + 'compare_{}.png'.format(tag))
    plt.close(0)
