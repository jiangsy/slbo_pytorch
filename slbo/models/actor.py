import torch
import torch.nn as nn
from typing import List, Callable, Optional

from slbo.models.initializer import normc_init
from slbo.models.utils import MLP, init
from slbo.models.actor_layer import *


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_space, hidden_dims: List[int],
                 state_normalizer: Optional[nn.Module]):
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


# class MLPActor(nn.Module):
#     def __init__(self, state_dim: int, dim_action: int, hidden_dims: List[int],
#                  state_normalizer: nn.Module = None):
#         super(MLPActor, self).__init__()
#         self.state_dim = state_dim
#         self.action_dim = dim_action
#         self.hidden_dims = hidden_dims
#
#         self.actor = MLP(state_dim, dim_action, hidden_dims, activation='Tanh')
#         self.log_std = nn.Parameter(torch.zeros([1, dim_action]), requires_grad=True)
#         self.state_normalizer = state_normalizer or nn.Identity()
#
#         init_ = lambda m: init(m, normc_init, lambda x: nn.init.constant_(x, 0))
#         init_last_ = lambda m: init(m, lambda x: normc_init(x, 0.01), lambda x: nn.init.constant_(x, 0))
#         self.actor.init(init_, init_last_)
#
#     def act(self, state: torch.Tensor, reparameterize=False, deterministic=False):
#         state = self.state_normalizer(state)
#         action_mean = self.actor(state)
#         action_dist = torch.distributions.Normal(action_mean, self.log_std.exp())
#
#         log_prob = None
#         entropy = None
#
#         if deterministic:
#             action = action_mean
#         else:
#             if reparameterize:
#                 action = action_dist.rsample()
#                 log_prob = action_dist.log_prob(action).sum(-1, keepdim=True)
#                 entropy = action_dist.entropy().sum(-1, keepdim=True)
#             else:
#                 action = action_dist.sample()
#                 log_prob = action_dist.log_prob(action).sum(-1, keepdim=True)
#                 entropy = action_dist.entropy().sum(-1, keepdim=True)
#         return action, log_prob, entropy, action_mean, self.log_std, self.log_std.exp()
#
#     def evaluate_action(self, action, state):
#         state = self.state_normalizer(state)
#         action_mean = self.actor(state)
#         action_dist = torch.distributions.Normal(action_mean, self.log_std.exp())
#
#         log_prob = action_dist.log_prob(action).sum(-1, keepdim=True)
#         entropy = action_dist.entropy().sum(-1, keepdim=True)
#
#         return log_prob, entropy
