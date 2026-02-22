import jittor as jt
import jittor.nn as nn
from tqdm import tqdm
import numpy as np
import os

def knn(query, data, k=10):

    assert data.shape[1] == query.shape[1]

    # 使用Jittor实现cdist
    M = jt.sqrt(((query.unsqueeze(1) - data.unsqueeze(0)) ** 2).sum(dim=2))
    v, ind = M.topk(k, largest=False)

    return v, ind[:, 0:min(k, data.shape[0])].int64()


def sample_knn_labels(query_embd, y_query, prior_embd, labels, k=10, n_class=10, weighted=False):

    n_sample = query_embd.shape[0]
    _, neighbour_ind = knn(query_embd, prior_embd, k=k)

    # compute the label of nearest neighbours
    neighbour_label_distribution = labels[neighbour_ind]

    # append the label of query
    neighbour_label_distribution = jt.concat((neighbour_label_distribution, y_query.unsqueeze(1)), 1)

    # sampling a label from the k+1 labels (k neighbours and itself)
    sampled_indices = jt.randint(0, k+1, (n_sample,))
    sampled_labels = neighbour_label_distribution[jt.arange(n_sample), sampled_indices]

    # convert labels to bincount (row wise)
    y_one_hot_batch = nn.one_hot(neighbour_label_distribution, num_classes=n_class).float32()

    # neighbour_freq = torch.sum(y_one_hot_batch, dim=1)[torch.tensor([range(n_sample)]), sampled_labels]
    neighbour_freq = jt.sum(y_one_hot_batch, dim=1)[jt.arange(n_sample), sampled_labels]

    # normalize max count as weight
    if weighted:
        weights = neighbour_freq / jt.sum(neighbour_freq)
    else:
        weights = 1/ n_sample * jt.ones([n_sample])

    return sampled_labels, jt.squeeze(weights)


def knn_labels(neighbours, indices, k=5, n_class=101):

    n_sample = len(indices)

    # compute the label of nearest neighbours
    neighbour_label_distribution = jt.array(neighbours[indices, :k+1]).int64()

    # sampling a label from the k+1 labels (k neighbours and itself)
    sampled_indices = jt.randint(0, k+1, (n_sample,))
    sampled_labels = neighbour_label_distribution[jt.arange(n_sample), sampled_indices]

    # convert labels to bincount (row wise)
    y_one_hot_batch = nn.one_hot(neighbour_label_distribution, num_classes=n_class).float32()

    neighbour_freq = jt.sum(y_one_hot_batch, dim=1)[jt.arange(n_sample), sampled_labels]

    # normalize max count as weight
    weights = neighbour_freq / jt.sum(neighbour_freq)

    return sampled_labels, jt.squeeze(weights)


def mean_knn_labels(query_embd, y_query, prior_embd, labels, k=100, n_class=10):

    _, neighbour_ind = knn(query_embd, prior_embd, k=k)

    # compute the label of nearest neighbours
    neighbour_label_distribution = labels[neighbour_ind]

    # append the label of query
    neighbour_label_distribution = jt.concat((neighbour_label_distribution, y_query.unsqueeze(1)), 1)

    one_hot_labels = nn.one_hot(neighbour_label_distribution, num_classes=n_class)
    mean_labels = jt.sum(one_hot_labels, dim=1) / (k+1)

    return mean_labels


def prepare_knn(labels, train_embed, save_dir, k=10):

    if os.path.exists(save_dir):
        neighbours = jt.array(np.load(save_dir))
        print(f'knn were computed before, loaded from: {save_dir}')
    else:
        neighbours = jt.zeros([train_embed.shape[0], k + 1]).int64()
        for i in tqdm(range(int(train_embed.shape[0] / 1000) + 1), desc='Searching knn', ncols=100):
            start = i * 1000
            end = min((i + 1) * 1000, train_embed.shape[0])
            ebd = train_embed[start:end, :]
            _, neighbour_ind = knn(ebd, train_embed, k=k + 1)
            neighbours[start:end, :] = labels[neighbour_ind]
        np.save(save_dir, neighbours)

    return neighbours
