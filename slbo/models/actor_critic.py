from typing import List

import numpy as np

from slbo.models.actor_layer import *
from slbo.models.utils import MLP, init


class ActorCritic(nn.Module):

    def __init__(self, dim_state, action_space, actor_hidden_dims: List[int], critic_hidden_dims: List[int],
                 normalizer: nn.Module = None):
        super(ActorCritic, self).__init__()

        self.actor_feature = MLP(dim_state, actor_hidden_dims[-1], actor_hidden_dims[:-1],
                                 activation='Tanh', last_activation='Tanh')
        self.critic = MLP(dim_state, 1, critic_hidden_dims, activation='Tanh', last_activation='Identity')
        self.normalizer = normalizer or nn.Identity()

        init_ = lambda m: init(m, lambda x: nn.init.orthogonal_(x, np.sqrt(2)), lambda x: nn.init.constant_(x, 0))
        self.actor_feature.init(init_, init_)
        self.critic.init(init_, init_)

        self.train()

        if action_space.__class__.__name__ == "Discrete":
            dim_action = action_space.n
            self.actor = CategoricalActorLayer(actor_hidden_dims[-1], dim_action)
        elif action_space.__class__.__name__ == "Box":
            dim_action = action_space.shape[0]
            self.actor = GaussianActorLayer(actor_hidden_dims[-1], dim_action, use_state_dependent_std=False)
        elif action_space.__class__.__name__ == "MultiBinary":
            dim_action = action_space.shape[0]
            self.actor = BernoulliActorLayer(actor_hidden_dims[-1], dim_action)

    def act(self, states, deterministic=False, reparamterize=False):
        action_feature, value = self.actor_feature(states), self.critic(states)
        action_dist, *_ = self.actor(action_feature)

        if deterministic:
            action = action_dist.mode()
        else:
            if reparamterize:
                action = action_dist.rsample()
            else:
                action = action_dist.sample()

        action_log_prob = action_dist.log_probs(action)
        dist_entropy = action_dist.entropy().mean()

        return value, action, action_log_prob, dist_entropy

    def criticize(self, states):
        values = self.critic(states)
        return values

    def evaluate_action(self, state, action):
        action_feature, value = self.actor_feature(state), self.critic(state)
        action_dist = self.actor(action_feature)

        action_log_probs = action_dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = action_dist.entropy().mean()

        return value, action_log_probs, dist_entropy

