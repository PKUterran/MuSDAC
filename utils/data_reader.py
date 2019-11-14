import math
import torch
import numpy as np
import scipy.sparse as sp
from sklearn import metrics


def encode_onehot(labels):
    classes = list(set(labels))
    classes = sorted(classes)
    print('classes:', classes)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path, dataset, n_meta=1):
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}/{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    adjs = []
    for meta in range(n_meta):
        edges_unordered = np.genfromtxt("{}/{}_meta_{}.cites".format(path, dataset, meta + 1),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.full(edges.shape[0], 1.), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = sp.coo_matrix(adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj))
        node_weight = [0.0] * len(idx)
        for i in range(0, len(adj.data)):
            node_weight[adj.row[i]] += adj.data[i]

        adj = normalize(adj + sp.eye(adj.shape[0]))
        D = sp.coo_matrix(
            [[1 / max(math.sqrt(node_weight[j]), 1e-6) if j == i else 0 for j in range(len(idx))] for i in range(len(idx))])
        adj = adj.todense()
        D = D.todense()
        adj = D * adj * D
        adj = torch.tensor(adj, dtype=torch.float32, requires_grad=False)
        adjs.append(adj)

    features = normalize(features)
    features = torch.tensor(np.array(features.todense()), dtype=torch.float32)
    labels = torch.tensor(np.where(labels)[1], dtype=torch.int64)

    return adjs, features, labels


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def macro_F1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    return metrics.f1_score(labels, preds, average='macro')
