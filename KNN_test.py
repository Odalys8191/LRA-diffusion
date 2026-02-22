import numpy as np
import jittor
import torchvision
from utils.data_utils import *
from utils.clip_wrapper import clip_img_wrap
from utils.model_SimCLR import SimCLR_encoder
import torchvision.transforms as transforms
import jittor.dataset as data
import argparse
from tqdm import tqdm


def knn(query, data, train_labels, k=10):

    assert data.shape[1] == query.shape[1]

    # sim_matrix = jittor.matmul(query, data.transpose())

    sim_matrix = jittor.cdist(query, data)
    # sim_weight, ind = sim_matrix.topk(k, dim=-1)

    sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1, largest=False)

    # print(sim_indices[:10, :])

    sim_labels = train_labels[sim_indices].squeeze()
    # sim_labels = jittor.gather(train_labels.expand(data.shape[0], -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / 0.5).exp()

    n_class = jittor.unique(train_labels).shape[0]

    # counts for each class
    one_hot_label = jittor.zeros(query.shape[0] * k, n_class)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = jittor.sum(one_hot_label.view(query.shape[0], -1, n_class) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)[:, :1].squeeze()

    return pred_labels

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, help="batch_size", type=int)
    parser.add_argument("--device", default='cpu', help="which GPU to use", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--k", default=10, help="num_workers", type=int)
    parser.add_argument("--data", default='cifar10', help="which dataset (cifar10 or cifar100)", type=str)
    parser.add_argument("--fp_encoder", default='SimCLR', help="which encoder (SimCLR or CLIP)", type=str)
    args = parser.parse_args()

    # load datasets
    if args.data == 'cifar10':
        n_class = 10
        train_dataset_cifar = torchvision.datasets.CIFAR10(root='../', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR10(root='../', train=False, download=True)
        # state_dict = jittor.load('../model/SimCLR_128_cifar10.pt')
    elif args.data == 'cifar100':
        n_class = 100
        train_dataset_cifar = torchvision.datasets.CIFAR100(root='../', train=True, download=True)
        test_dataset_cifar = torchvision.datasets.CIFAR100(root='../', train=False, download=True)
    else:
        raise Exception("Date should be cifar10 or cifar100")

    # load fp encoder
    if args.fp_encoder == 'SimCLR':
        fp_dim = 2048
        # state_dict = jittor.load(f'../model/SimCLR_128_{args.data}.pt')
        encoder_model = SimCLR_encoder(feature_dim=128)
        # encoder_model.load_state_dict(state_dict, strict=False)
    elif args.fp_encoder == 'CLIP':
        encoder_model = clip_img_wrap('ViT-L/14', args.device)
        fp_dim = encoder_model.dim
    else:
        raise Exception("fp_encoder should be SimCLR or CLIP")

    train_dataset = Custom_dataset(train_dataset_cifar.data[:45000], train_dataset_cifar.targets[:45000])
    test_dataset = Custom_dataset(test_dataset_cifar.data, test_dataset_cifar.targets)

    train_labels = jittor.array(train_dataset.targets).squeeze()
    test_labels = jittor.array(test_dataset.targets).squeeze()

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

    # compute embedding fp(x) for training and testing set
    with jittor.no_grad():

        train_embed = []
        for i, (images, labels, indices) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=150):
            train_embed.append(encoder_model(images))

        train_embed = jittor.concat(train_embed, dim=0)

        test_embed = []
        for i, (images, labels, indices) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=150):
            feature = encoder_model(images)
            test_embed.append(feature)
        test_embed = jittor.concat(test_embed, dim=0)

    pred_labels = knn(test_embed, train_embed, train_labels, k=args.k)
    acc = jittor.sum(pred_labels == test_labels)
    acc = acc / pred_labels.shape[0]
    print(f'KNN accuracy: {100 * acc.item():.2f}, feature space: {args.fp_encoder}')
