import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class OnPolicyBuffer(object):
    def __init__(self, num_steps, num_envs, obs_shape, action_space,
                 use_gae=True, gamma=0.99, gae_lambda=0.95, use_proper_time_limits=True):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.states = torch.zeros(num_steps + 1, num_envs, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.values = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.bad_masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

        self.use_gae = use_gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits

    def to(self, device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, states, actions, action_log_probs,
               values, rewards, masks, bad_masks):
        self.states[self.step + 1].copy_(states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.values[self.step].copy_(values)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value):
        if self.use_proper_time_limits:
            if self.use_gae:
                self.values[-1] = next_value
                gae = 0
                for step in reversed(range(self.num_steps)):
                    delta = self.rewards[step] + self.gamma * self.values[step + 1] * self.masks[step + 1] - \
                            self.values[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.values[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.num_steps)):
                    self.returns[step] = (self.returns[step + 1] *
                        self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * self.values[step]
        else:
            if self.use_gae:
                self.values[-1] = next_value
                gae = 0
                for step in reversed(range(self.num_steps)):
                    delta = self.rewards[step] + self.gamma * self.values[step + 1] * self.masks[step + 1] - self.values[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    self.returns[step] = gae + self.values[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.num_steps)):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def get_batch_generator(self, batch_size=None, advantages=None):
        batch_size = self.num_steps * self.num_envs if batch_size is None else batch_size
        sampler = BatchSampler(SubsetRandomSampler(range(self.num_steps * self.num_envs)), batch_size, drop_last=True)

        for indices in sampler:
            states = self.states[:-1].view(-1, *self.states.size()[2:])[indices]
            actions = self.actions.view(-1, self.actions.size(-1))[indices]
            values = self.values[:-1].view(-1, 1)[indices]
            returns = self.returns[:-1].view(-1, 1)[indices]
            masks = self.masks[:-1].view(-1, 1)[indices]
            action_log_probs = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targets = None
            else:
                adv_targets = advantages.view(-1, 1)[indices]

            yield {'states': states, 'actions': actions, 'values': values, 'returns': returns,
                   'masks': masks, 'action_log_probs': action_log_probs, 'adv_targets': adv_targets}
