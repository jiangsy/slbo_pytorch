from typing import List

import torch
import torch.nn as nn

from slbo.models.initializer import truncated_norm_init
from slbo.models.normalizers import Normalizers
from slbo.models.utils import MLP, init


class Dynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], normalizer: Normalizers):
        super(Dynamics, self).__init__()
        self.dim_state = state_dim
        self.dim_action = action_dim
        self.normalizer = normalizer
        self.diff_dynamics = MLP(state_dim + action_dim, state_dim, hidden_dims, activation='ReLU')

        init_ = lambda m: init(m, truncated_norm_init, lambda x: nn.init.constant_(x, 0))
        self.diff_dynamics.init(init_, init_)

    def forward(self, state, action):
        # action clip is the best normalization according to the authors
        x = torch.cat([self.normalizer.state_normalizer(state), action.clamp(-1., 1.)], dim=-1)
        normalized_diff = self.diff_dynamics(x)
        next_states = state + self.normalizer.diff_normalizer(normalized_diff, inverse=True)
        next_states = self.normalizer.state_normalizer(self.normalizer.state_normalizer(next_states).clamp(-100, 100),
                                                       inverse=True)
        return next_states


