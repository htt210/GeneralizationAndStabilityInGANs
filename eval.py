import torch
from inception_score import inception_score


class Evaluator(object):
    def __init__(self, generator, nz, batch_size=64,
                 inception_nsamples=60000, device='cuda'):
        self.generator = generator
        self.nz = nz
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size
        self.device = device

    def compute_inception_score(self):
        self.generator.eval()
        imgs = []
        print('start')
        while(len(imgs) < self.inception_nsamples):
            ztest = torch.randn(self.batch_size, self.nz, 1, 1, device=self.device)

            samples = self.generator(ztest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)
        print('done collecting samples')
        imgs = imgs[:self.inception_nsamples]
        score, score_std = inception_score(
            imgs, device=self.device, resize=True, splits=10
        )
        print('done calculating')

        return score, score_std

    def create_samples(self, z):
        self.generator.eval()

        # Sample x
        with torch.no_grad():
            x = self.generator(z)
        return x
