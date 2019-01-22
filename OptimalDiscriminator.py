import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as ag
from torch.autograd import Variable

import os
import random

device = 'cuda'

np.random.seed(0)

means = np.array([[0.4, 0.9],
                  [1.3, 1.2]])
vars = np.array([[[1, 0.5],
                  [0.5, 1]],
                 [[1, 0.7],
                  [0.7, 1]]])

n_points = 100
LOWER = -2.5
UPPER = 4.5
x = y = np.linspace(LOWER, UPPER, n_points)
n_samples = 100
# n_train = n_samples * 2
batch_size = 20
n_batches = n_samples // batch_size
lr = 1e-3
center = 0
LAMBDA = 1
alpha = None
ones = torch.ones(batch_size, device=device)
zeros = torch.zeros(batch_size, device=device)

root = "./figs/"
prefix = root + "optimalD_linear_gradpen_center_%.2f_alpha_%s_lambda_%.2f_nsamples_%d/" % (center, str(alpha), LAMBDA, n_samples)
if not os.path.exists(prefix):
    os.mkdir(prefix)

real = np.random.multivariate_normal(means[0], vars[0], n_samples)
fake = np.random.multivariate_normal(means[1], vars[1], n_samples)
real_data = torch.tensor(real, device=device, dtype=torch.float)
fake_data = torch.tensor(fake, device=device, dtype=torch.float)
# labels = torch.Tensor(np.array([1] * n_samples + [0] * n_samples))
# data = torch.Tensor(np.concatenate([pos, neg], axis=0))

points = np.zeros((n_points, n_points, 2), dtype='float32')
points[:, :, 0] = np.linspace(LOWER, UPPER, n_points)[:, None]
points[:, :, 1] = np.linspace(LOWER, UPPER, n_points)[None, :]
points = points.reshape((-1, 2))
points = torch.tensor(points, device=device)


def calc_gradient_penalty(netD, real_data, fake_data, center=0, alpha=None, LAMBDA=.5, device=None):
    #print real_data.size()
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


def density_ratio(x, mean_real, var_real, mean_fake, var_fake):
    # print('start')
    real = multivariate_normal(mean=mean_real, cov=var_real)
    fake = multivariate_normal(mean=mean_fake, cov=var_fake)

    pdf_real = real.pdf(x)
    pdf_fake = fake.pdf(x)
    # print('pdf_real', pdf_real)

    return pdf_real / (pdf_real + pdf_fake)


D = nn.Sequential(nn.Linear(2, 64),
                  nn.Tanh(),
                  nn.Linear(64, 64),
                  nn.Tanh(),
                  nn.Linear(64, 1),
                  nn.Sigmoid())
# D = nn.Sequential(nn.Linear(2, 1),
#                   nn.Sigmoid())
D.to(device=device)

criterion = nn.BCELoss()
optimizer = optim.Adam(D.parameters(), lr=1e-3)

fig0, ax0 = plt.subplots(1, 1, figsize=(5, 4))
fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4))

for epoch in range(0, 10001):
    real_idx = list(range(n_samples))
    fake_idx = list(range(n_samples))
    random.shuffle(real_idx)
    random.shuffle(fake_idx)

    for bidx in range(n_batches):
        optimizer.zero_grad()

        real_batch = real_data[real_idx[bidx * batch_size: (bidx + 1) * batch_size], :]
        predict_real = D(real_batch)
        # target_real = density_ratio(real_batch.cpu().data.numpy(), means[0], vars[0], means[1], vars[1])
        # target_real = torch.tensor(target_real, device=device, dtype=torch.float)
        # loss_real = criterion.forward(predict_real, target_real)
        loss_real = criterion.forward(predict_real, ones)
        loss_real.backward()

        fake_batch = fake_data[fake_idx[bidx * batch_size: (bidx + 1) * batch_size], :]
        predict_fake = D(fake_batch)
        # target_fake = density_ratio(fake_batch.cpu().data.numpy(), means[0], vars[0], means[1], vars[1])
        # target_fake = torch.tensor(target_fake, device=device, dtype=torch.float)
        # loss_fake = criterion.forward(predict_fake, target_fake)
        loss_fake = criterion.forward(predict_fake, zeros)
        loss_fake.backward()

        grad_pen = calc_gradient_penalty(D, real_batch.detach(), fake_batch.detach(), center=center, alpha=alpha, LAMBDA=LAMBDA, device=device)
        grad_pen.backward()

        loss = loss_real + loss_fake

        optimizer.step()

    if epoch % 1000 == 0:
        print('loss %d:' % epoch, loss)
        print('gradpen %d:' % epoch, grad_pen)
        ax0.clear()
        point_values = D(points)
        point_values = point_values.data.cpu().numpy().reshape(n_points, n_points).transpose()
        cplot = ax0.contourf(x, y, point_values, cmap='hot', alpha=0.7, levels=np.arange(0, 1.01, 0.1))
        ax0.scatter(real[:, 0], real[:, 1], c='r', marker='+')
        ax0.scatter(fake[:, 0], fake[:, 1], c='b', marker='o')

        # ax[1].clear()
        optimal_values = density_ratio(points.data.cpu().numpy(), means[0], vars[0], means[1], vars[1])
        optimal_values = optimal_values.reshape(n_points, n_points).transpose()
        # cplot = ax[1].contourf(x, y, optimal_values, cmap='hot', alpha=0.7, levels=np.arange(0, 1.01, 0.1))
        # ax[1].scatter(real[:, 0], real[:, 1], c='r', marker='+')
        # ax[1].scatter(fake[:, 0], fake[:, 1], c='b', marker='o')

        # plt.clabel(cplot, fontsize=9, inline=1)
        # if epoch == 0:
        #     cbar_ax = fig0.add_axes([0.93, 0.15, 0.02, 0.7])
        #     fig0.colorbar(cplot, cax=cbar_ax)
        fig0.savefig(prefix + 'optimal_discriminator_%04d.pdf' % epoch, bbox_inches='tight')

        ax1[0].imshow(np.rot90(point_values.transpose(), 1))
        ax1[1].imshow(np.rot90(optimal_values.transpose(), 1))

        plt.pause(0.1)

    # print(idx)

plt.show()


