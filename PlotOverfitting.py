import numpy as np
import matplotlib.pyplot as plt
import torch
import re

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
model1 = {'filename': './figs/gan_mnist_center_0.00_alpha_None_lambda_100.00_lrg_0.00030_lrd_0.00090_nhidden_512_nz_50_ncritic_5/loss.txt',
         'label': 'GAN-0-GP'}
model2 = {'filename': './figs/gan_mnist_center_0.00_alpha_1.0_lambda_100.00_lrg_0.00030_lrd_0.00090_nhidden_512_nz_50_ncritic_5/loss.txt',
         'label': 'GAN-0-real'}
model3 = {'filename': './figs/gan_mnist_center_0.00_alpha_1.0_lambda_0.00_lrg_0.00030_lrd_0.00030_nhidden_512_nz_50_ncritic_1/loss.txt',
          'label': 'GAN'}
model4 = {'filename': './figs/gan_mnist_center_1.00_alpha_None_lambda_100.00_lrg_0.00030_lrd_0.00030_nhidden_512_nz_50_ncritic_1/loss.txt',
          'label': 'GAN-1-GP'}


def parse_file(filename):
    losses = []
    correct = []
    with open(filename, 'r') as f:
        for line in f:
            line = re.sub('[^\d\.]', ' ', line).split()
            # print(line)
            if len(line) > 0:
                losses.append(float(line[0]))
                correct.append(int(line[1]))

    print(losses)
    print(correct)
    return np.array(losses[:10]), np.array(correct[:10]) / 1000.


loss1, correct1 = parse_file(model1['filename'])
loss2, correct2 = parse_file(model2['filename'])
loss3, correct3 = parse_file(model3['filename'])
loss4, correct4 = parse_file(model4['filename'])

ax[0].plot(loss1[1:], label=model1['label'], linestyle=':')
ax[0].plot(loss2[1:], label=model2['label'], linestyle='--')
ax[0].plot(loss3[1:], label=model3['label'], linestyle='-.')
ax[0].plot(loss4[1:], label=model4['label'], linestyle='-')
ax[0].set_xlabel('Iterations (x1000)')
ax[0].set_ylabel('Loss')
# ax[0].legend()

ax[1].plot(correct1[1:], label=model1['label'], linestyle=':')
ax[1].plot(correct2[1:], label=model2['label'], linestyle='--')
ax[1].plot(correct3[1:], label=model3['label'], linestyle='-.')
ax[1].plot(correct4[1:], label=model4['label'], linestyle='-')
ax[1].set_xlabel('Iteartions (x1000)')
ax[1].set_ylabel('Accuracy')
# ax[1].legend()

plt.legend()
fig.savefig('figs/overfitting.pdf', bbox_inches='tight')
plt.show()
