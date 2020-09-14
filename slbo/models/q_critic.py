import torch
import torch.nn as nn
import numpy as np

from slbo.models.utils import MLP, init
from slbo.models.initializer import fanin_init


class QCritic(nn.Module):
    def __init__(self, state_dim, action_space, hidden_dims, activation='ReLU', last_activation='Identity',
                 init_w=3e-3, init_b=0.1, use_multihead_output=False):
        super(QCritic, self).__init__()

        assert not use_multihead_output or action_space.__class__.__name__ == 'Discrete'

        if action_space.__class__.__name__ == 'Discrete':
            action_dim = action_space.n
        elif action_space.__class__.__name__ == 'Box':
            action_dim = action_space.shape[0]

        if use_multihead_output:
            action_dim = action_space.n
            self.critic = MLP(state_dim, action_dim, hidden_dims,
                              activation=activation, last_activation=last_activation)
            self.forward = self._get_q_value_discrete
        else:
            self.critic = MLP(state_dim + action_dim, 1, hidden_dims,
                              activation=activation, last_activation=last_activation)
            self.forward = self._get_q_value_continuous

        init_ = lambda m: init(m, fanin_init, lambda x: nn.init.constant_(x, init_b))
        init_last_ = lambda m: init(m, lambda x: nn.init.uniform_(x, -init_w, init_w),
                                       lambda x: nn.init.uniform_(x, -init_w, init_w))
        self.critic.init(init_, init_last_)

    def _get_q_value_continuous(self, state, action):
        return self.critic(torch.cat([state, action], dim=-1))

    def _get_q_value_discrete(self, state, action):
        return self.critic_feature(state)[action]
