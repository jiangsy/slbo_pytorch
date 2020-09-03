import torch
import torch.nn as nn
from typing import List, Callable, Optional

from slbo.models.initializer import normc_init
from slbo.models.utils import MLP, init
from slbo.models.actor_layer import *


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_space, hidden_dims: List[int],
                 state_normalizer: Optional[nn.Module], use_limited_entropy=False):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_space
        self.hidden_dims = hidden_dims

        self.actor_feature = MLP(state_dim, hidden_dims[-1], hidden_dims[:-1],
                                 activation='Tanh', last_activation='Tanh')
        self.state_normalizer = state_normalizer or nn.Identity()

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.actor = CategoricalActorLayer(hidden_dims[-1], action_dim)
        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            if use_limited_entropy:
                self.actor = LimitedEntGaussianActorLayer(hidden_dims[-1], action_dim, use_state_dependent_std=False)
            else:
                self.actor = GaussianActorLayer(hidden_dims[-1], action_dim, use_state_dependent_std=False)
        elif action_space.__class__.__name__ == "MultiBinary":
            action_dim = action_space.shape[0]
            self.actor = BernoulliActorLayer(hidden_dims[-1], action_dim)
        else:
            raise NotImplemented

        init_ = lambda m: init(m, normc_init, lambda x: nn.init.constant_(x, 0))
        self.actor_feature.init(init_, init_)

    def act(self, states, deterministic=False, reparamterize=False):
        states = self.state_normalizer(states)
        action_features = self.actor_feature(states)
        action_dists, action_means, log_stds = self.actor(action_features)

        if deterministic:
            actions = action_dists.mode()
        else:
            if reparamterize:
                actions = action_dists.rsample()
            else:
                actions = action_dists.sample()

        log_probs = action_dists.log_probs(actions)
        entropy = action_dists.entropy().mean()

        return actions, log_probs, entropy, action_means, log_stds, log_stds.exp()

    def evaluate_action(self, states, actions):
        states = self.state_normalizer(states)
        action_feature = self.actor_feature(states)
        action_dist, *_ = self.actor(action_feature)

        log_probs = action_dist.log_probs(actions)
        entropy = action_dist.entropy().mean()

        return log_probs, entropy
