from typing import List

import torch
import torch.nn as nn

from slbo.models.normalizer import Normalizer
from slbo.models.utils import MLP


class DensityEstimator(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], state_action_ratio: bool,
                 normalizer: Normalizer = None, last_activation='Identity'):
        super(DensityEstimator, self).__init__()
        self.dim_state = state_dim
        self.dim_action = action_dim
        self.normalizer = normalizer

        self.state_action_ratio = state_action_ratio

        if state_action_ratio:
            self.density_estimator = MLP(state_dim + action_dim, 1, hidden_dims, activation='ReLU',
                                         last_activation=last_activation)
        else:
            self.density_estimator = MLP(state_dim, 1, hidden_dims, activation='ReLU', last_activation=last_activation)

        # init_ = lambda m: init(m, truncated_norm_init, lambda x: nn.init.constant_(x, 0))
        # self.diff_dynamics.init(init_, init_)

    def forward(self, state, action):
        if self.state_action_ratio:
            x = torch.cat([self.normalizer(state), action.clamp(-1., 1.)], dim=-1)
        else:
            x = self.normalizer(state)
        # action clip is the best normalization according to the authors
        ratio = self.density_estimator(x)
        return ratio


