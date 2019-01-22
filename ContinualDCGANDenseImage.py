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
import datetime
import torchvision
from Classifier import Net
import torchvision.datasets as dset
from eval import Evaluator


class Generator(nn.Module):
    def __init__(self, nz, ngf):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        return self.net(z)


class ContinualClassifier(nn.Module):
    def __init__(self, nc, ndf, discount):
        super(ContinualClassifier, self).__init__()

        self.net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.task_count = 0
        self.running_fisher_info = None
        self.discount = discount

        self.prev_params = {}
        for n, p in self.named_parameters():
            self.prev_params[n] = p.detach().clone()

    def forward(self, x):
        return self.net(x)

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
        alpha = torch.rand(real_data.size(0), 1, 1, 1, device=device)
    # print(alpha.size())
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


def evaluate(model, dataset='stacked', device='cuda'):
    model.eval()
    batch_size = 100
    criterion = nn.BCELoss()
    ones = torch.ones(batch_size, device=device)

    if dataset == 'mnist':
        data = MNISTDataset(train=False)
    elif dataset == 'stacked':
        data = StackedMNISTDataset(train=False)

    n_batches = data.n_samples // batch_size
    test_loss = torch.tensor(0., device=device)
    correct = torch.tensor(0, device=device)
    with torch.no_grad():
        for i in range(n_batches):
            real_batch = data.next_batch(batch_size, device=device)
            # print('real_batch.size', real_batch.size())
            predict_real = D(real_batch)
            # print(predict_real)
            # print(i, (predict_real > 0.5).sum())
            correct += (predict_real > 0.5).sum()
            test_loss += criterion.forward(predict_real, ones).sum()

    test_loss /= data.n_samples
    accuracy = float(correct) / data.n_samples
    return test_loss, correct, accuracy


def count_modes(G, classifier, noise, nc=3, img_size=28, batch_size=100, n_samples=100000, device='cuda'):
    classifier.to(device)
    n_batches = n_samples // batch_size
    bins = torch.zeros(10, 10, 10).to(device)
    with torch.no_grad():
        for i in range(n_batches):
            noise_batch = noise.next_batch(batch_size, device=device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.view(batch_size * nc, 1, img_size, img_size)
            output = F.softmax(classifier(fake_batch))
            # if output[0][4:].sum() > 0.5:
            #     print(output[0])
            output[output < 0.5] = 0
            predict = output.max(1, keepdim=True)[1].long()
            # print(predict[predict > 3])
            # print('-----------')
            # print(predict)
            bins[predict[::3], predict[1::3], predict[2::3]] += 1

    n_modes = (bins > 10).sum()
    print('n_modes', n_modes)
    return n_modes


def GAN_GP(D, G, data, noise, nc, img_size, nepochs=10000, d_steps=1, batch_size=32,
           lrg=1e-3, lrd=3e-3, center=0, LAMBDA=1, alpha=None,
           device='cuda', prefix='figs/', args=None):
    D.to(device)
    G.to(device)
    optim_d = optim.Adam(D.parameters(), lr=lrd, betas=(0.5, 0.999))
    optim_g = optim.Adam(G.parameters(), lr=lrg, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    zeros = torch.zeros(batch_size, device=device)
    ones = torch.ones(batch_size, device=device)

    fixed_z = noise.next_batch(64, device=device)
    inter_z = torch.tensor(fixed_z.data, device=device)

    for i in range(8):
        for j in range(8):
            inter_j = j / 8.0
            inter_z[i * 8 + j] = (1 - inter_j) * fixed_z[i * 2] + inter_j * fixed_z[i * 2 + 1]

    logf = open(prefix + '/inception.txt', 'w')

    start = datetime.datetime.now()
    for epoch in range(nepochs):
        for i, real_batch in enumerate(data):
            if i % 10 == 0:
                print('iter', i)
                if args.debug:
                    break

            # train D
            optim_d.zero_grad()
            real_batch = real_batch[0].to(device)
            predict_real = D(real_batch)
            loss_real = criterion.forward(predict_real, ones)

            noise_batch = noise.next_batch(batch_size, device=device)
            fake_batch = G(noise_batch)
            fake_batch = fake_batch.detach()
            predict_fake = D(fake_batch)
            loss_fake = criterion.forward(predict_fake, zeros)

            loss_d = loss_real + loss_fake

            if LAMBDA > 0.:
                gradpen = cal_gradpen(D, real_batch.detach(), fake_batch.detach(),
                                      center=center, LAMBDA=LAMBDA, alpha=alpha, device=device)
                loss_d += gradpen

            if args.ewcweight > 0.:
                D._compute_fisher(real_batch, fake_batch, ones, zeros, args.batch_mode)
                loss_ewc = D.ewc_loss()
                loss_d += loss_ewc

            loss_d.backward()
            optim_d.step()

            # train G
            if i % d_steps == d_steps - 1:
                optim_g.zero_grad()
                noise_batch = noise.next_batch(batch_size, device=device)
                fake_batch = G(noise_batch)
                predict_fake = D(fake_batch)
                loss_g = criterion.forward(predict_fake, ones)
                loss_g.backward()
                optim_g.step()

        print(datetime.datetime.now() - start, 'epoch %d' % epoch)
        if epoch % args.inceptioninterval == 0:
            evaluator = Evaluator(G, args.nz, device=device)
            inception_score, inception_score_std = evaluator.compute_inception_score()
            print(datetime.datetime.now() - start, 'inception %d: %f %f \n' % (epoch, inception_score, inception_score_std))
            logf.write('%d, %.4f, %.4f \n' % (epoch, inception_score, inception_score_std))
            logf.flush()
            sample_imgs = evaluator.create_samples(fixed_z)
            torchvision.utils.save_image(sample_imgs, prefix + '/epoch_%d.png' % epoch)

    return D, G


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters')
    parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
    parser.add_argument('--device', type=str, default='cuda', help='id of the gpu. -1 for cpu')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--center', type=float, default=0., help='gradpen center')
    parser.add_argument('--LAMBDA', type=float, default=0., help='gradpen weight')
    parser.add_argument('--alpha', type=float, default=None,
                        help='interpolation weight between reals and fakes. '
                             '1 for real only, 0 for fake only, None for random interpolation')
    parser.add_argument('--lrg', type=float, default=2e-4, help='lr for G')
    parser.add_argument('--lrd', type=float, default=2e-4, help='lr for D')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use: cifar10')
    parser.add_argument('--loss', type=str, default='gan', help='gan | wgan')
    parser.add_argument('--nz', type=int, default=100, help='dimensionality of noise')
    parser.add_argument('--ncritic', type=int, default=1,
                        help='number of critic/discriminator iterations per generator iteration')
    parser.add_argument('--batch_mode', type=bool, default=True,
                        help='compute Fisher inforation in batch or for each individual sample')
    parser.add_argument('--discount', type=float, default=0.999, help='discount factor for old Fisher info')
    parser.add_argument('--ewcweight', type=float, default=0., help='weight of the ewc loss term')
    parser.add_argument('--inceptioninterval', type=int, default=10, help='calculate inception every n epochs')
    parser.add_argument('--debug', type=bool, default=False, help='run debug mode')

    args = parser.parse_args()
    nz = args.nz
    if args.dataset == 'cifar10':
        nc = 3
        img_size = 64
        dataset = dset.CIFAR10(root='data/cifar10', download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
        data = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=int(4), drop_last=True)

    prefix = 'figs/dc%s_%s_center_%.2f_alpha_%s_lambda_%.2f_lrg_%.5f_lrd_%.5f_nz_%d_discount' \
             '_%.4f_batch_%d_ewcweight_%.2f_ndf_%d_ngf_%d_ncritic_%d/' % \
             (args.loss, args.dataset, args.center, str(args.alpha), args.LAMBDA, args.lrg, args.lrd,
              args.nz, args.discount, args.batch_mode, args.ewcweight, args.ndf, args.ngf, args.ncritic)
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    print(prefix)

    G = Generator(nz, args.ngf)
    if args.loss == 'gan':
        D = ContinualClassifier(nc, args.ndf, args.discount)

    print('D')
    print(D)
    noise = NoiseDataset(dim=nz, is_image=True)

    config = str(args) + '\n' + str(G) + '\n' + str(D) + '\n'
    with open(prefix + 'config.txt', 'w') as f:
        f.write(config)

    if args.loss == 'gan':
        print(args.loss)
        D, G = GAN_GP(D, G, data, noise, nc=nc, img_size=img_size, nepochs=args.nepochs+1, d_steps=args.ncritic,
                      batch_size=args.batch_size, lrg=args.lrg, lrd=args.lrd, center=args.center, LAMBDA=args.LAMBDA,
                      alpha=args.alpha, device=args.device, prefix=prefix, args=args)
    # else:
    #     print(args.loss)
    #     D, G = WGAN_GP(D, G, data, noise, nc=nc, img_size=img_size, niter=args.nepochs, d_steps=args.ncritic,
    #                    batch_size=args.batch_size, lrg=args.lrg, lrd=args.lrd, center=args.center,
    #                    LAMBDA=args.LAMBDA, alpha=args.alpha, device=args.device, prefix=prefix, args=args)

    torch.save(D, prefix + 'D.t7')
    torch.save(G, prefix + 'G.t7')