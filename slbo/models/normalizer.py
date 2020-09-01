from typing import List

import torch
import torch.distributions.kl as kl
import torch.nn as nn
from stable_baselines import logger


class GaussianNormalizer(nn.Module):
    def __init__(self, shape: List[int], eps=1e-8, verbose=0):  # batch_size x ...
        super().__init__()

        self.shape = shape
        self.verbose = verbose

        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.std = torch.ones(shape, dtype=torch.float32)
        self.eps = eps
        self.n = 0

    def forward(self, x: torch.Tensor, inverse=False):
        if inverse:
            return x * self.std + self.mean
        return (x - self.mean) / (torch.clamp(self.std, min=self.eps))

    def to(self, *args, **kwargs):
        self.mean = self.mean.to(*args, **kwargs)
        self.std = self.std.to(*args, **kwargs)

    # noinspection DuplicatedCode
    def update(self, samples: torch.Tensor):
        old_mean, old_std, old_n = self.mean, self.std, self.n
        samples = samples - old_mean
        n = samples.shape[0]
        delta = samples.mean(dim=0)
        new_n = old_n + n
        new_mean = old_mean + delta * n / new_n
        new_std = torch.sqrt((old_std**2 * old_n + samples.var(dim=0) * n + delta**2 * old_n * n / new_n) / new_n)
        kl_old_new = kl.kl_divergence(torch.distributions.Normal(new_mean, new_std),
                                      torch.distributions.Normal(old_mean, old_std)).sum()
        self.mean, self.std, self.n = new_mean, new_std, new_n

        if self.verbose > 0:
            logger.info("updating Normalizer<%s>, KL divergence = %.6f", self.name, kl_old_new)

    # noinspection PyMethodOverriding
    def state_dict(self, *args, **kwargs):
        return {'mean': self.mean, 'std': self.std, 'n': self.n}

    # noinspection PyMethodOverriding
    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']
        self.n = state_dict['n']


class Normalizer(nn.Module):
    def __init__(self, dim_action: int, dim_state: int, verbose=0):
        super().__init__()
        # action_normalizer is not used
        self.action_normalizer = GaussianNormalizer([dim_action], verbose=verbose)
        self.state_normalizer = GaussianNormalizer([dim_state], verbose=verbose)
        self.diff_normalizer = GaussianNormalizer([dim_state], verbose=verbose)

    def forward(self):
        raise NotImplemented

    def to(self, *args, **kwargs):
        self.action_normalizer.to(*args, **kwargs)
        self.state_normalizer.to(*args, **kwargs)
        self.diff_normalizer.to(*args, **kwargs)

    # noinspection PyMethodOverriding
    def state_dict(self, *args, **kwargs):
        return {'action_normalizer': self.action_normalizer.state_dict(),
                'state_normalizer': self.state_normalizer.state_dict(),
                'diff_normalizer': self.diff_normalizer.state_dict()}

    # noinspection PyMethodOverriding, PyTypeChecker
    def load_state_dict(self, state_dict):
        self.action_normalizer.load_state_dict(state_dict['action_normalizer'])
        self.state_normalizer.load_state_dict(state_dict['state_normalizer'])
        self.diff_normalizer.load_state_dict(state_dict['diff_normalizer'])



