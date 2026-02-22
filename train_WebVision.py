import numpy as np
import jittor.dataset as data
from tqdm import tqdm
from utils.ema import EMA
from utils.clip_wrapper import clip_img_wrap
from utils.webvision_data_utils import WebVision
import jittor.optim as optim
from utils.learning import *
from model_diffusion import Diffusion
from utils.knn_utils import sample_knn_labels, knn, knn_labels, prepare_knn
import argparse
import random
jittor.manual_seed(123)
np.random.seed(123)
random.seed(123)


def train(diffusion_model, val_loader, device, model_save_dir, args, data_dir):
    # device = diffusion_model.device

    k = args.k
    n_epochs = args.nepoch
    n_class = 50

    val_embed = np.load(os.path.join(data_dir, 'fp_embed_val_webvision.npy'))
    train_embed = jittor.array(np.load(os.path.join(data_dir, 'fp_embed_train_webvision.npy')))
    train_labels = jittor.array(np.load(os.path.join(data_dir, 'train_labels_webvision.npy')))

    diffusion_model.fp_encoder.eval()
    params = list(diffusion_model.model.parameters()) + list(diffusion_model.diffusion_encoder.parameters())
    optimizer = optim.Adam(params, lr=0.0001, weight_decay=0.0, betas=(0.9, 0.999),
                           eps=1e-08)
    diffusion_loss = nn.MSELoss(reduction='none')
    # diffusion_loss = nn.MSELoss()
    ema_helper = EMA(mu=0.9999)
    ema_helper.register(diffusion_model.model)

    max_accuracy = 0

    print('Diffusion training start')
    for epoch in range(n_epochs):
        train_dataset = WebVision(data_root=data_dir, split='train', balance=True, randomize=True, cls_size=500, transform='train')
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers, drop_last=True)
        diffusion_model.diffusion_encoder.train()
        diffusion_model.model.train()

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train diffusion epoch {epoch}', ncols=120) as pbar:
            for i, (x_batch, y_batch, _) in pbar:

                with jittor.no_grad():
                    fp_embd = diffusion_model.fp_encoder(x_batch)

                y_labels_batch, sample_weight = sample_knn_labels(fp_embd, y_batch, train_embed,
                                                                  train_labels, k=k, n_class=n_class)

                y_one_hot_batch, y_logits_batch = cast_label_to_one_hot_and_prototype(y_labels_batch.int64(),
                                                                                      n_class=n_class)
                y_0_batch = y_one_hot_batch

                # adjust_learning_rate
                adjust_learning_rate(optimizer, i / len(train_loader) + epoch, warmup_epochs=1, n_epochs=n_epochs, lr_input=0.001)
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

                # loss = diffusion_loss(e, output)
                pbar.set_postfix({'loss': loss.item()})

                # optimize diffusion model that predicts eps_theta
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(diffusion_model.model.parameters(), 1.0)
                nn.utils.clip_grad_norm_(diffusion_model.diffusion_encoder.parameters(), 1.0)
                optimizer.step()
                ema_helper.update(diffusion_model.model)

        acc_val = test(diffusion_model, val_loader, val_embed)
        print(f"epoch: {epoch}, val accuracy: {acc_val:.2f}%")
        if acc_val > max_accuracy:
            if args.device is None:
                states = [diffusion_model.model.module.state_dict(),
                          diffusion_model.diffusion_encoder.module.state_dict()]
            else:
                states = [diffusion_model.model.state_dict(),
                          diffusion_model.diffusion_encoder.state_dict()]
            jittor.save(states, model_save_dir)
            print("Model saved, best test accuracy at Epoch {}.".format(epoch))
            max_accuracy = max(max_accuracy, acc_val)


def test(diffusion_model, test_loader, test_embed):

    if not isinstance(test_embed, jittor.Var):
        test_embed = jittor.array(test_embed).float32()

    correct_cnt = 0
    with jittor.no_grad():
        diffusion_model.model.eval()
        diffusion_model.diffusion_encoder.eval()
        diffusion_model.fp_encoder.eval()
        for test_batch_idx, data_batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f'evaluating diff', ncols=100):
            [x_batch, target, indicies] = data_batch[:3]
            fp_embed = test_embed[indicies, :]
            label_t_0 = diffusion_model.reverse_ddim(x_batch, stochastic=False, fp_x=fp_embed).detach()
            # acc_temp = accuracy(label_t_0.detach(), target)[0].item()
            # acc_avg += acc_temp

            correct = cnt_agree(label_t_0.detach(), target)
            correct_cnt += correct

    acc = 100 * correct_cnt / test_embed.shape[0]
    return acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nepoch", default=300, help="number of training epochs", type=int)
    parser.add_argument("--batch_size", default=256, help="batch_size", type=int)
    parser.add_argument("--num_workers", default=16, help="num_workers", type=int)
    parser.add_argument("--warmup_epochs", default=1, help="warmup_epochs", type=int)
    parser.add_argument("--feature_dim", default=1024, help="feature_dim", type=int)
    parser.add_argument("--k", default=50, help="k neighbors for knn", type=int)
    parser.add_argument("--ddim_n_step", default=10, help="number of steps in ddim", type=int)
    parser.add_argument("--diff_encoder", default='resnet50_l', help="which encoder for diffusion", type=str)
    parser.add_argument("--gpu_devices", default=[0, 1, 2, 3], type=int, nargs='+', help="")
    parser.add_argument("--device", default=None, help="which cuda to use", type=str)
    args = parser.parse_args()

    if args.device is None:
        gpu_devices = ','.join([str(id) for id in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        device = 'cuda' if jittor.has_cuda else 'cpu'
    else:
        device = args.device

    n_class = 50

    # load datasets WebVsison
    webvision_dir = os.path.join(os.getcwd(), 'WebVision')
    print('data_dir', webvision_dir)

    train_dataset = WebVision(data_root=webvision_dir, split='train', balance=False, randomize=False, cls_size=500,
                              transform='val')
    train_labels = jittor.array(train_dataset.targets).long()
    np.save(os.path.join(webvision_dir, f'train_labels_webvision.npy'), train_labels.numpy())
    val_dataset = WebVision(data_root=webvision_dir, split='val')
    val_loader = data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=args.num_workers)

    # initialize diffusion model
    fp_encoder = clip_img_wrap('ViT-L/14', device, center=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    fp_dim = fp_encoder.dim
    model_path = './model/LRA-diffusion_WebVision.pt'
    diffusion_model = Diffusion(fp_encoder, num_timesteps=1000, n_class=n_class, fp_dim=fp_dim, device=device,
                                feature_dim=args.feature_dim, encoder_type=args.diff_encoder,
                                ddim_num_steps=args.ddim_n_step, beta_schedule='cosine')
    # state_dict = jittor.load(model_path)
    # diffusion_model.load_diffusion_net(state_dict)
    diffusion_model.fp_encoder.eval()

    # DataParallel wrapper
    if args.device is None:
        print('using DataParallel')
        diffusion_model.model = nn.DataParallel(diffusion_model.model)
        diffusion_model.diffusion_encoder = nn.DataParallel(diffusion_model.diffusion_encoder)
        diffusion_model.fp_encoder = nn.DataParallel(fp_encoder)
    else:
        print('using single gpu')

    # # pre-compute for fp embeddings on training data
    print('pre-computing fp embeddings for training data')
    train_embed_dir = os.path.join(webvision_dir, f'fp_embed_train_webvision.npy')
    train_embed = prepare_fp_x(diffusion_model.fp_encoder, train_dataset, train_embed_dir, device=device,
                               fp_dim=fp_dim, batch_size=200)
    # for validation data
    print('pre-computing fp embeddings for validation data for webvision')
    val_embed_dir = os.path.join(webvision_dir, f'fp_embed_val_webvision.npy')
    val_embed = prepare_fp_x(diffusion_model.fp_encoder, val_dataset, val_embed_dir, device=device,
                             fp_dim=fp_dim, batch_size=200)

    max_accuracy = test(diffusion_model, val_loader, val_embed)
    print('test webvision accuracy:', max_accuracy)

    # train the diffusion model
    train(diffusion_model, val_loader, device, model_path, args, data_dir=webvision_dir)





