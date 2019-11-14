import numpy as np
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import warnings

from .data_reader import encode_onehot
warnings.filterwarnings('ignore')


def write_files(file_tag, feas_a, feas_b, labels_a, labels_b, directory=''):
    file = open(directory + 'a_emb_' + file_tag + '.txt', 'w')
    for item in feas_a:
        for i in range(len(item)):
            file.write(str(item[i]))
            file.write(' ')
        file.write('\n')
    file.close()
    file = open(directory + 'a_label_' + file_tag + '.txt', 'w')
    for item in labels_a:
        file.write(str(item) + '\n')
    file.close()
    file = open(directory + 'b_emb_' + file_tag + '.txt', 'w')
    for item in feas_b:
        for i in range(len(item)):
            file.write(str(item[i]))
            file.write(' ')
        file.write('\n')
    file.close()
    file = open(directory + 'b_label_' + file_tag + '.txt', 'w')
    for item in labels_b:
        file.write(str(item) + '\n')
    file.close()


def classify(tag='0', comparison=False, directory=''):
    colorlist = ['green', 'red', 'black', 'blue', 'yellow', 'pink', 'purple', 'grey']
    f = open(directory + 'a_emb_' + tag + '.txt', 'r')
    feasA = []
    for line in f:
        strtmp = line.strip().split(' ')
        feasA.append([float(strtmp[i]) for i in range(len(strtmp))])
    feasA = np.array(feasA)
    f = open(directory + 'a_label_' + tag + '.txt', 'r')
    labelA = []
    for line in f:
        labelA.append(float(line))
    labelA = np.array(labelA)
    colorA = [colorlist[int(labelA[i])] for i in range(len(labelA))]

    f = open(directory + 'b_emb_' + tag + '.txt', 'r')
    feasB = []
    for line in f:
        strtmp = line.strip().split(' ')
        feasB.append([float(strtmp[i]) for i in range(len(strtmp))])
    feasB = np.array(feasB)
    f = open(directory + 'b_label_' + tag + '.txt', 'r')
    labelB = []
    for line in f:
        labelB.append(float(line))
    labelB = np.array(labelB)
    colorB = [colorlist[int(labelB[i])] for i in range(len(labelB))]

    fig = plt.figure(figsize=(8, 8))

    tsne = TSNE(n_components=2)
    X = np.vstack((feasA, feasB))
    print(X[0].size)
    transed = tsne.fit_transform(X)
    pickle.dump(transed, open(directory + '2-D_' + tag, 'wb'))
    trans_feasA = transed[0:len(feasA)]
    print("A finish")
    trans_feasB = transed[len(feasA):]
    plt.scatter(trans_feasA[:, 0], trans_feasA[:, 1], c='blue', marker='*', cmap=plt.cm.Spectral)
    plt.scatter(trans_feasB[:, 0], trans_feasB[:, 1], c='black', marker='o', cmap=plt.cm.Spectral)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.savefig(directory + 'data_visual_sep-init_' + tag + '.png')
    plt.close(0)

    fig = plt.figure(figsize=(8, 8))
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.scatter(trans_feasB[:, 0], trans_feasB[:, 1], c=colorB, marker='o', cmap=plt.cm.Spectral)
    plt.savefig(directory + 'data_visualB-init_' + tag + '.png')
    plt.close(0)

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(trans_feasA[:, 0], trans_feasA[:, 1], c=colorA, marker='*', cmap=plt.cm.Spectral)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.savefig(directory + 'data_visualA-init_' + tag + '.png')
    plt.close(0)

    if not comparison:
        return
    if tag == '0':
        return
    # lines
    no_gan_results = []
    test_results = []
    for line in open(directory + 'no_gan_result.txt'):
        no_gan_results.append(json.loads(line))
    for line in open(directory + 'test_result_' + tag + '.txt'):
        test_results.append(json.loads(line))
    no_gan_result = {
        'epoch': [i['epoch'] for i in no_gan_results],
        'f1_src': [i['f1_a'] for i in no_gan_results],
        'f1_tgt': [i['f1_b'] for i in no_gan_results]
    }
    test_result = {
        'epoch': [i['epoch'] for i in test_results],
        'f1_src': [i['f1_a'] for i in test_results],
        'f1_tgt': [i['f1_b'] for i in test_results]
    }
    fig = plt.figure(figsize=(8, 8))
    plt.plot(no_gan_result['epoch'], no_gan_result['f1_src'], c='blue', linestyle='--')
    plt.plot(no_gan_result['epoch'], no_gan_result['f1_tgt'], c='blue')
    plt.plot(test_result['epoch'], test_result['f1_src'], c='red', linestyle='--')
    plt.plot(test_result['epoch'], test_result['f1_tgt'], c='red')
    plt.xlabel('epochs', fontsize=12)
    plt.savefig(directory + 'tendency_' + str(tag) + '.png')
    plt.close(0)


if __name__ == '__main__':
    classify('1000')
