from typing import List

import torch
import torch.nn as nn

from slbo.models.initializer import normc_init
from slbo.models.utils import MLP, init


class QCritic(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, hidden_states: List[int]):
        super(QCritic, self).__init__()
        self.critic = MLP(dim_state + dim_action, hidden_states, 1)

    def forward(self, state, action):
            x = torch.cat([state, action], dim=-1)
            return self.critic(x)


class VCritic(nn.Module):
    def __init__(self, dim_state: int, hidden_dims: List[int], state_normalizer=None, activation='Tanh'):
        super(VCritic, self).__init__()
        self.critic = MLP(dim_state, 1, hidden_dims, activation=activation)
        self.normalizer = state_normalizer or nn.Identity()

        init_ = lambda m: init(m, normc_init, lambda x: nn.init.constant_(x, 0))
        self.critic.init(init_, init_)

    def forward(self, state):
        state = self.normalizer(state)
        return self.critic(state)
