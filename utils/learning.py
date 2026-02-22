import random
import math
import numpy as np
import jittor as jt
import jittor.nn as nn
import os
from tqdm import tqdm


def set_random_seed(seed):
    print(f"\n* Set seed {seed}")
    jt.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def adjust_learning_rate(optimizer, epoch, warmup_epochs=40, n_epochs=1000, lr_input=0.001):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        lr = lr_input * epoch / warmup_epochs
    else:
        lr = 0.0 + lr_input * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (n_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def cast_label_to_one_hot_and_prototype(y_labels_batch, n_class, return_prototype=True):
    """
    y_labels_batch: a vector of length batch_size.
    """
    y_one_hot_batch = nn.one_hot(y_labels_batch, num_classes=n_class).float32()
    if return_prototype:
        label_min, label_max = [0.001, 0.999]
        y_logits_batch = jt.logit(nn.normalize(
            jt.clamp(y_one_hot_batch, min=label_min, max=label_max), p=1.0, dim=1))
        return y_one_hot_batch, y_logits_batch
    else:
        return y_one_hot_batch


# random seed related
def init_fn(worker_id):
    np.random.seed(77 + worker_id)


def prepare_fp_x(fp_encoder, dataset, save_dir=None, device='cpu', fp_dim=768, batch_size=400):

    if save_dir is not None:
        if os.path.exists(save_dir):
            fp_embed_all = jt.array(np.load(save_dir))
            print(f'Embeddings were computed before, loaded from: {save_dir}')
            return fp_embed_all

    with jt.no_grad():
        # 使用Jittor的DataLoader
        data_loader = dataset.set_attrs(batch_size=batch_size, shuffle=False, num_workers=4)
        fp_embed_all = jt.zeros([len(dataset), fp_dim])
        with tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Computing embeddings fp(x)',
                  ncols=100) as pbar:
            for i, data_batch in pbar:
                [x_batch, _, data_indecies] = data_batch[:3]
                temp = fp_encoder(x_batch)
                fp_embed_all[data_indecies, :] = temp

        if save_dir is not None:
            np.save(save_dir, fp_embed_all.numpy())

    return fp_embed_all


def cnt_agree(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    maxk = min(max(topk), output.shape[1])

    output = nn.softmax(-(output - 1)**2,  dim=-1)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.transpose(0, 1)
    correct = pred.equal(target.reshape(1, -1).expand_as(pred))

    return jt.sum(correct).item()


