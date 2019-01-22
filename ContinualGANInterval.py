import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as ag
import argparse
from Datasets import *
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, nhidden, nlayers):
        super(Generator, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('input', nn.Linear(2, nhidden))
        self.net.add_module('act0', nn.ReLU(True))
        for i in range(nlayers):
            self.net.add_module('hidden_%d' % (i + 1), nn.Linear(nhidden, nhidden))
            self.net.add_module('act_%d' % (i + 1), nn.ReLU(True))
        self.net.add_module('output', nn.Linear(nhidden, 2))

    def forward(self, z):
        return self.net(z)


class ContinualClassifier(nn.Module):
    def __init__(self, nhidden, nlayers, discount):
        super(ContinualClassifier, self).__init__()
        self.net = nn.Sequential()
        self.net.add_module('input', nn.Linear(2, nhidden))
        self.net.add_module('act0', nn.ReLU(True))
        for i in range(nlayers):
            self.net.add_module('hidden_%d' % (i + 1), nn.Linear(nhidden, nhidden))
            self.net.add_module('act_%d' % (i + 1), nn.ReLU(True))
        self.net.add_module('output', nn.Linear(nhidden, 1))
        self.net.add_module('sigmoid', nn.Sigmoid())

        self.task_count = 0
        self.running_fisher_info = None
        self.discount = discount

        self.prev_params = {}
        for n, p in self.named_parameters():
            self.prev_params[n] = p.detach().clone()

    def forward(self, x):
        return self.net(x)

    def compute_fisher(self, real_list, fake_list, ones, zeros, batch_mode=True):
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                est_fisher_info[n] = p.detach().clone().to(p.device).zero_()
        # Set model to evaluation mode
        mode = self.training
        self.eval()

        for reals, fakes in zip(real_list, fake_list):
            if batch_mode:
                # batch mode
                self.zero_grad()
                out_real = self.net(reals)
                loss_real = F.binary_cross_entropy(out_real, ones)
                loss_real.backward()
                out_fake = self.net(fakes)
                loss_fake = F.binary_cross_entropy(out_fake, zeros)
                loss_fake.backward()

                # Square gradients and keep running sum
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        if p.grad is not None:
                            est_fisher_info[n] += p.grad.detach() ** 2
            else:
                data = [*reals.split(1), *fakes.split(1)]
                n_samples = len(data)
                labels = torch.zeros(n_samples, 1, device=self._device())
                labels[0:n_samples // 2] = 1
                labels = labels.split(1)
                for datum, label in zip(data, labels):
                    self.zero_grad()
                    # print('datum', datum)
                    output = self.net(datum)
                    negloglikelihood = F.binary_cross_entropy(output, label)
                    negloglikelihood.backward()

                    # Square gradients and keep running sum
                    for n, p in self.named_parameters():
                        if p.requires_grad:
                            if p.grad is not None:
                                est_fisher_info[n] += p.grad.detach() ** 2

                est_fisher_info = {n: p / n_samples for n, p in est_fisher_info.items()}

        est_fisher_info = {n: p / len(real_list) for n, p in est_fisher_info.items()}

        self.train(mode=mode)
        self.task_count = 1

        # update the running Fisher information
        if self.running_fisher_info is None:
            self.running_fisher_info = est_fisher_info
        for n, p in self.named_parameters():
            if p.requires_grad:
                self.prev_params[n] = p.detach().clone()
                self.running_fisher_info[n].mul_(self.discount).add_(est_fisher_info[n] * (1 - self.discount))

    def _compute_fisher(self, reals, fakes, ones, zeros, batch_mode=True):
        est_fisher_info = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                est_fisher_info[n] = p.detach().clone().to(p.device).zero_()
        # Set model to evaluation mode
        mode = self.training
        self.eval()

        if batch_mode:
            # batch mode
            self.zero_grad()
            out_real = self.net(reals)
            loss_real = F.binary_cross_entropy(out_real, ones)
            loss_real.backward()
            out_fake = self.net(fakes)
            loss_fake = F.binary_cross_entropy(out_fake, zeros)
            loss_fake.backward()

            # Square gradients and keep running sum
            for n, p in self.named_parameters():
                if p.requires_grad:
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2
        else:
            data = [*reals.split(1), *fakes.split(1)]
            n_samples = len(data)
            labels = torch.zeros(n_samples, 1, device=self._device())
            labels[0:n_samples // 2] = 1
            labels = labels.split(1)
            for datum, label in zip(data, labels):
                self.zero_grad()
                # print('datum', datum)
                output = self.net(datum)
                negloglikelihood = F.binary_cross_entropy(output, label)
                negloglikelihood.backward()

                # Square gradients and keep running sum
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        if p.grad is not None:
                            est_fisher_info[n] += p.grad.detach() ** 2

            est_fisher_info = {n: p / n_samples for n, p in est_fisher_info.items()}

        self.train(mode=mode)
        self.task_count = 1

        # update the running Fisher information
        if self.running_fisher_info is None:
            self.running_fisher_info = est_fisher_info
        for n, p in self.named_parameters():
            if p.requires_grad:
                self.prev_params[n] = p.detach().clone()
                self.running_fisher_info[n].mul_(self.discount).add_(est_fisher_info[n] * (1 - self.discount))

    def ewc_loss(self):
        if self.task_count>0:
            losses = []
            # If "offline EWC", loop over all previous tasks (if "online EWC", [EWC_task_count]=1 so only 1 iteration)
            for task in range(1, self.task_count+1):
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        # Retrieve stored mode (MAP estimate) and precision (Fisher Information matrix)
                        mean = self.prev_params[n]
                        fisher = self.running_fisher_info[n]
                        # Calculate EWC-loss
                        losses.append((fisher * (p-mean)**2).sum())
            # Sum EWC-loss from all parameters (and from all tasks, if "offline EWC")
            return (1./2)*sum(losses)
        else:
            # EWC-loss is 0 if there are no stored mode and precision yet
            return torch.tensor(0., device=self._device())

    def _device(self):
        return next(self.parameters()).device


def cal_gradpen(netD, real_data, fake_data, center=0, alpha=None, LAMBDA=1, device=None):
    if alpha is not None:
        alpha = torch.tensor(alpha, device=device)  # torch.rand(real_data.size(0), 1, device=device)
    else:
        alpha = torch.rand(real_data.size(0), 1, device=device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = ag.grad(outputs=disc_interpolates, inputs=interpolates,
                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                        create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - center) ** 2).mean() * LAMBDA
    return gradient_penalty


def visualize_grad(G: Generator, D: ContinualClassifier, criterion, fig, ax, scale, device=None):
    nticks = 20
    noise_batch = (torch.rand(nticks * nticks, 2, device=device) - 0.5) * 4
    ones = torch.ones(nticks * nticks, 1, device=device)

    step = 2 * scale / nticks
    for i in range(nticks):
        for j in range(nticks):
            noise_batch[i * nticks + j, 0] = -scale + i * step
            noise_batch[i * nticks + j, 1] = -scale + j * step

    noise_batch.requires_grad_()
    with torch.enable_grad():
        out_batch = D(noise_batch)
        if isinstance(D, ContinualClassifier):
            loss = criterion.forward(out_batch, ones)
            loss.backward()
        else:
            loss = -out_batch.mean()
            loss.backward()

    coord = noise_batch.data.cpu().numpy()
    grad = -noise_batch.grad.cpu().numpy()

    ax.quiver(coord[:, 0], coord[:, 1], grad[:, 0], grad[:, 1])
    return coord, grad


def accumulate_grad(acc_grad, new_grad, new_grad_weight=0.01):
    if acc_grad is None:
        acc_grad = new_grad
    grad = new_grad * new_grad_weight + acc_grad * (1 - new_grad_weight)
    return grad


def show_grad(coord, grad, fig, ax):
    ax.quiver(coord[:, 0], coord[:, 1], grad[:, 0], grad[:, 1])


def GAN_GP(D: ContinualClassifier, G: Generator, data, noise, niter=10000, batch_size=32, optimizer='Adam',
           lrg=1e-3, lrd=3e-3, center=0, LAMBDA=1, alpha=None, device='cuda', prefix='figs/', args=None):
    D.to(device)
    G.to(device)

    if optimizer == 'SGD':
        optim_d = optim.SGD(D.parameters(), lr=lrd)
        optim_g = optim.SGD(G.parameters(), lr=lrg)
    elif optimizer == 'Adam':
        optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.9))
        optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.9))

    criterion = nn.BCELoss()

    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)

    real_buffer = []
    fake_buffer = []

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # acc_fig, acc_ax = plt.subplots(1, 1, figsize=(4, 4))
    # acc_grad = None

    scale = 1
    if args is not None:
        scale = args.scale
    scale *= data.range

    for iter in range(niter):
        if iter % 100 == 0:
            print(iter)

            noise_batch = noise.next_batch(512, device=device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.data.cpu().numpy()

            real_batch = data.next_batch(512, device=device)

            plt.figure(fig.number)
            ax.clear()
            ax.scatter(real_batch[:, 0], real_batch[:, 1], s=2)
            ax.scatter(fake_batch[:, 0], fake_batch[:, 1], s=2, c='r', marker='+')
            ax.set_xlim((-scale, scale))
            ax.set_ylim((-scale, scale))

            coord, grad = visualize_grad(G, D, criterion, fig, ax, scale=scale, device=device)
            plt.draw()
            plt.savefig(prefix + 'fig_%05d.pdf' % iter, bbox_inches='tight')
            plt.pause(0.1)

        # train D
        optim_d.zero_grad()
        real_batch = data.next_batch(batch_size, device=device)
        predict_real = D(real_batch)
        loss_real = criterion.forward(predict_real, ones)

        noise_batch = noise.next_batch(batch_size, device=device)
        fake_batch = G(noise_batch)
        fake_batch = fake_batch.detach()
        predict_fake = D(fake_batch)
        loss_fake = criterion.forward(predict_fake, zeros)

        gradpen = cal_gradpen(D, real_batch.detach(), fake_batch.detach(),
                              center=center, LAMBDA=LAMBDA, alpha=alpha, device=device)

        D._compute_fisher(real_batch, fake_batch, ones, zeros, args.batch_mode)

        # real_buffer.append(real_batch)
        # fake_buffer.append(fake_batch)
        # if iter % args.ewcinterval == args.ewcinterval - 1:
        #     D.compute_fisher(real_buffer, fake_buffer, ones, zeros, args.batch_mode)
        #     real_buffer = []
        #     fake_buffer = []

        loss_ewc = D.ewc_loss()

        loss_d = loss_real + loss_fake + gradpen + args.ewcweight * loss_ewc
        loss_d.backward()
        optim_d.step()

        # train G
        optim_g.zero_grad()
        noise_batch = noise.next_batch(batch_size, device=device)
        fake_batch = G(noise_batch)
        predict_fake = D(fake_batch)
        loss_g = criterion.forward(predict_fake, ones)
        loss_g.backward()
        optim_g.step()

    return D, G


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nhidden', type=int, default=64, help='number of hidden neurons')
    parser.add_argument('--gnlayers', type=int, default=2, help='number of hidden layers in generator')
    parser.add_argument('--dnlayers', type=int, default=1, help='number of hidden layers in discriminator/critic')
    parser.add_argument('--gradweight', type=float, default=0.01, help='weight of the new grad in the moving average')
    parser.add_argument('--batch_mode', type=bool, default=True,
                        help='compute Fisher inforation in batch or for each individual sample')
    parser.add_argument('--discount', type=float, default=0.9, help='discount factor for old Fisher info')
    parser.add_argument('--ewcweight', type=float, default=10., help='weight of the ewc loss term')
    parser.add_argument('--ewcinterval', type=int, default=10, help='interval for recomputing ewcloss')
    parser.add_argument('--niters', type=int, default=20000, help='number of iterations')
    parser.add_argument('--device', type=str, default='cuda', help='id of the gpu. -1 for cpu')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--center', type=float, default=0., help='gradpen center')
    parser.add_argument('--LAMBDA', type=float, default=0., help='gradpen weight')
    parser.add_argument('--alpha', type=float, default=None, help='interpolation weight between reals and fakes')
    parser.add_argument('--lrg', type=float, default=3e-3, help='lr for G')
    parser.add_argument('--lrd', type=float, default=3e-3, help='lr for D')
    parser.add_argument('--dataset', type=str, default='8Gaussians',
                        help='dataset to use: 8Gaussians | 25Gaussians | swissroll')
    parser.add_argument('--scale', type=float, default=2., help='data scaling')
    parser.add_argument('--loss', type=str, default='gan', help='gan | wgan')
    parser.add_argument('--optim', type=str, default='SGD', help='optimizer to use')
    parser.add_argument('--ncritic', type=int, default=1, help='critic iters / generator iter')

    args = parser.parse_args()

    prefix = 'figs/continual_%s_%s_gradfield_center_%.2f_alpha_%s_lambda_%.2f_lrg_%.5f_lrd_%.5f_nhidden_%d_scale_%.2f' \
             '_optim_%s_gnlayers_%d_dnlayers_%d_gradweight_%.4f_discount_%.4f_batch_%d_ewcweight_%.2f_ewcinterval_%4d' \
             '_ncritic_%d/' % \
             (args.loss, args.dataset, args.center, str(args.alpha), args.LAMBDA, args.lrg, args.lrd, args.nhidden,
              args.scale, args.optim, args.gnlayers, args.dnlayers, args.gradweight, args.discount, args.batch_mode,
              args.ewcweight, args.ewcinterval, args.ncritic)
    print(prefix)
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    G = Generator(args.nhidden, args.gnlayers)
    if args.loss == 'gan':
        D = ContinualClassifier(args.nhidden, args.dnlayers, args.discount)
    # else:
    #     D = Critic(args.nhidden, args.dnlayers)
    noise = NoiseDataset()
    data = ToyDataset(distr=args.dataset, scale=args.scale)
    if args.loss == 'gan':
        print(args.loss)
        GAN_GP(D, G, data, noise, niter=args.niters+1, batch_size=args.batch_size, optimizer=args.optim,
               lrg=args.lrg, lrd=args.lrd, center=args.center, LAMBDA=args.LAMBDA, alpha=args.alpha,
               device=args.device, prefix=prefix, args=args)
    # elif args.loss == 'wgan':
    #     print(args.loss)
    #     WGAN_GP(D, G, data, noise, niter=args.niters + 1, ncritic=args.ncritic, batch_size=args.batch_size,
    #             optimizer=args.optim, lrg=args.lrg, lrd=args.lrd, center=args.center, LAMBDA=args.LAMBDA,
    #             alpha=args.alpha, device=args.device, prefix=prefix, args=args)
    # elif args.loss == 'wganlp':
    #     print(args.loss)
    #     WGAN_LP(D, G, data, noise, niter=args.niters + 1, ncritic=args.ncritic, batch_size=args.batch_size,
    #             optimizer=args.optim, lrg=args.lrg, lrd=args.lrd, center=args.center, LAMBDA=args.LAMBDA,
    #             alpha=args.alpha, device=args.device, prefix=prefix, args=args)

