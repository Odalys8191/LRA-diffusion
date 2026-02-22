import os.path
import jittor.nn as nn
import jittor.dataset as data
from tqdm import tqdm
from utils.ema import EMA
import numpy as np
import random
from utils.clip_wrapper import clip_img_wrap
from utils.food_data_utils import Food101N, gen_train_list, gen_test_list
import jittor
import jittor.optim as optim
from utils.learning import *
from model_diffusion import Diffusion
from utils.knn_utils import knn_labels, knn, prepare_knn
import argparse
jittor.manual_seed(123)
np.random.seed(123)
random.seed(123)


def train(diffusion_model, train_loader, test_loader, model_save_dir, n_epochs=1000, knn=10, data_dir='./Food101N_data'):
    device = diffusion_model.device
    n_class = diffusion_model.n_class

    clip_test_embed = np.load(os.path.join(data_dir, 'fp_embed_test_food.npy'))
    clip_train_embed = np.load(os.path.join(data_dir, 'fp_embed_train_food.npy'))
    neighbours = np.load(os.path.join(data_dir, 'fp_knn_food.npy'))

    optimizer = optim.Adam(diffusion_model.model.parameters(), lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')

    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)

    max_accuracy = 0.0
    print('Diffusion training start')
    for epoch in range(n_epochs):
        diffusion_model.model.train()

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, (x_batch, _, data_indices) in pbar:

                fp_embd = jittor.array(clip_train_embed[data_indices, :]).float32()
                y_labels_batch, sample_weight = knn_labels(neighbours, data_indices, k=knn, n_class=101)

                y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch.int64(),
                                                                                      n_class=n_class)
                y_0_batch = y_one_hot_batch

                # adjust_learning_rate
                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=10, n_epochs=200, lr_input=0.001)
                n = x_batch.shape[0]

                # antithetic sampling
                t = jittor.randint(low=0, high=diffusion_model.num_timesteps, shape=(n // 2 + 1,))
                t = jittor.concat([t, diffusion_model.num_timesteps - 1 - t], dim=0)[:n]

                # train with and without prior
                output, e = diffusion_model.forward_t(y_0_batch, x_batch, t, fp_embd)

                # compute loss
                mse_loss = diffusion_loss(e, output)
                weighted_mse_loss = jittor.matmul(sample_weight, mse_loss)
                loss = weighted_mse_loss.mean()

                pbar.set_postfix({'loss': loss.item()})

                # optimize diffusion model that predicts eps_theta
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model.model)

        acc_test = test(diffusion_model, test_loader, clip_test_embed)
        print(f"epoch: {epoch}, diff accuracy: {acc_test:.2f}%")
        if acc_test > max_accuracy:
            # save diffusion model
            if args.device is None:
                states = [diffusion_model.model.module.state_dict(),
                          diffusion_model.diffusion_encoder.module.state_dict()]
            else:
                states = [diffusion_model.model.state_dict(),
                          diffusion_model.diffusion_encoder.state_dict()]
            jittor.save(states, model_save_dir)
            print("Model saved, update best accuracy at Epoch {}.".format(epoch))
            max_accuracy = max(max_accuracy, acc_test)


def test(diffusion_model, test_loader, test_embed):

    if not isinstance(test_embed, jittor.Var):
        test_embed = jittor.array(test_embed).float32()

    with jittor.no_grad():
        diffusion_model.model.eval()
        diffusion_model.fp_encoder.eval()
        correct_cnt = 0.
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'evaluating diff', ncols=100):
            [images, target, indicies] = data_batch[:3]

            fp_embed = test_embed[indicies, :]
            label_t_0 = diffusion_model.reverse_ddim(images, stochastic=False, fp_x=fp_embed).detach()
            correct = cnt_agree(label_t_0.detach(), target)[0].item()
            correct_cnt += correct

    acc = 100 * correct_cnt / test_embed.shape[0]
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", default=100, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=256, help="batch_size", type=int)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=1, help="warmup_epochs", type=int)
    parser.add_argument("--feature_dim", default=1024, help="feature_dim", type=int)
    parser.add_argument("--k", default=10, help="k neighbors for knn", type=int)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--diff_encoder", default='resnet50_l',
                        help="which encoder for diffusion (resnet18_l, 34_l, 50_l...)", type=str)
    parser.add_argument("--gpu_devices", default=[0, 1, 2, 3], type=int, nargs='+', help="")
    parser.add_argument("--device", default=None, help="which cuda to use", type=str)
    args = parser.parse_args()

    if args.device is None:
        gpu_devices = ','.join([str(id) for id in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        device = 'cuda' if jittor.has_cuda else 'cpu'
    else:
        device = args.device

    # load dataset
    data_dir = os.path.join(os.getcwd(), 'Food101N')
    batch_size = args.batch_size
    num_workers = args.num_workers
    n_class = 101

    # prepare dataset directories
    map_name2cat = gen_train_list(data_dir)
    gen_test_list(map_name2cat, data_dir)

    # create dataset and loader
    train_dataset = Food101N(data_path=data_dir, split='train')
    labels = jittor.array(train_dataset.targets)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
    test_dataset = Food101N(data_path=data_dir, split='test')
    test_loader = data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=num_workers)

    # initialize diffusion model
    fp_encoder = clip_img_wrap('ViT-L/14', device, center=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    fp_dim = fp_encoder.dim
    model_path = './model/LRA-diffusion_Food101N.pt'
    diffusion_model = Diffusion(fp_encoder, num_timesteps=1000, n_class=n_class, fp_dim=fp_dim, device=device,
                                feature_dim=args.feature_dim, encoder_type=args.diff_encoder,
                                ddim_num_steps=args.ddim_n_step)
    # # load trained checkpoint to do test.
    # state_dict = jittor.load(model_path)
    # diffusion_model.load_diffusion_net(state_dict)
    diffusion_model.fp_encoder.eval()

    # DataParallel wrapper
    if args.device is None:
        print('using DataParallel')
        diffusion_model.model = nn.DataParallel(diffusion_model.model)
        diffusion_model.diffusion_encoder = nn.DataParallel(diffusion_model.diffusion_encoder)
        diffusion_model.fp_encoder = nn.DataParallel(diffusion_model.fp_encoder)
        fp_encoder = nn.DataParallel(fp_encoder)
    else:
        print('using single gpu')

    # pre-compute for fp embeddings on training data
    print('pre-computing fp embeddings for training data')
    train_embed_dir = os.path.join(data_dir, 'fp_embed_train_food.npy')
    train_embed = prepare_fp_x(fp_encoder, train_dataset, train_embed_dir, device=device, fp_dim=fp_dim)
    # for testing data
    print('pre-computing fp embeddings for testing data')
    test_embed_dir = os.path.join(data_dir, 'fp_embed_test_food.npy')
    test_embed = prepare_fp_x(fp_encoder, test_dataset, test_embed_dir, device=device, fp_dim=fp_dim)

    # pre-compute knns on training data
    print('pre-compute knns on training data')
    neighbours = prepare_knn(labels, train_embed, os.path.join(data_dir, 'fp_knn_food.npy'), k=args.k)

    acc_diff = test(diffusion_model, test_loader, test_embed)
    print(acc_diff)

    # train the diffusion model
    train(diffusion_model, train_loader, test_loader, model_path, n_epochs=args.nepoch, knn=args.k, data_dir=data_dir)







