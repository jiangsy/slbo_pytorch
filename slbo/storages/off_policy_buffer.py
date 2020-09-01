import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class OffPolicyBuffer(object):
    def __init__(self, buffer_size, num_envs, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.states = torch.zeros(buffer_size, num_envs, state_dim)
        self.next_states = torch.zeros(buffer_size, num_envs, state_dim)
        self.rewards = torch.zeros(buffer_size, num_envs, 1)
        self.actions = torch.zeros(buffer_size, num_envs, action_dim)
        self.masks = torch.ones(buffer_size + 1, num_envs, 1)
        self.bad_masks = torch.ones(buffer_size + 1, num_envs, 1)

        self.buffer_size = buffer_size
        self.index = 0
        self.size = 0
        self.device = torch.device('cpu')

    def to(self, device):
        self.states = self.states.to(device)
        self.next_states = self.next_states.to(device)
        self.rewards = self.rewards.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

        self.device = device

    def add_buffer(self, buffer):
        for idx in range(buffer.size):
            self.insert(buffer.states[idx], buffer.actions[idx], buffer.rewards[idx], buffer.next_states[idx],
                        buffer.masks[idx], buffer.bad_masks[idx])

    def insert(self, states, actions, rewards, next_states, masks, bad_masks):
        self.states[self.index, :, :].copy_(states)
        self.actions[self.index, :, :].copy_(actions)
        self.rewards[self.index, :, :].copy_(rewards)
        self.next_states[self.index, :, :].copy_(next_states)
        self.masks[self.index, :, :].copy_(masks)
        self.bad_masks[self.index, :, :].copy_(bad_masks)

        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def clear(self):
        self.index = 0
        self.size = 0

    def get_batch_generator(self, batch_size):
        sampler = BatchSampler(SubsetRandomSampler(range(self.size * self.num_envs)), batch_size, drop_last=True)

        for indices in sampler:
            states = self.states.view(-1, *self.states.size()[2:])[indices]
            actions = self.actions.view(-1, self.actions.size(-1))[indices]
            rewards = self.rewards.view(-1, 1)[indices]
            next_states = self.next_states.view(-1, *self.states.size()[2:])[indices]
            masks = self.masks.view(-1, 1)[indices]
            bad_masks = self.bad_masks.view(-1, 1)[indices]

            yield {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states,
                   'masks': masks, 'bad_masks': bad_masks}

    def get_sequential_batch_generator(self, batch_size, num_steps):
        sampler = BatchSampler(SubsetRandomSampler(range(self.size - num_steps)),
                               int(batch_size / self.num_envs), drop_last=True)

        for indices in sampler:
            indices = np.array(indices)
            states = torch.zeros([batch_size, num_steps, self.states.shape[-1]], device=self.device)
            next_states = torch.zeros([batch_size, num_steps, self.next_states.shape[-1]], device=self.device)
            actions = torch.zeros([batch_size, num_steps, self.actions.shape[-1]], device=self.device)
            rewards = torch.zeros([batch_size, num_steps, 1], device=self.device)
            masks = torch.zeros([batch_size, num_steps, 1], device=self.device)
            bad_masks = torch.zeros([batch_size, num_steps, 1], device=self.device)
            for step in range(num_steps):
                states[:, step, :].copy_(self.states[indices + step].view(-1, self.states.shape[-1]))
                next_states[:, step, :].copy_(self.next_states[indices + step].view(-1, self.next_states.shape[-1]))
                actions[:, step, :].copy_(self.actions[indices + step].view(-1, self.actions.shape[-1]))
                rewards[:, step, :].copy_(self.rewards[indices + step].view(-1, 1))
                masks[:, step, :].copy_(self.masks[indices + step].view(-1, 1))
                bad_masks[:, step, :].copy_(self.bad_masks[indices + step].view(-1, 1))

            yield {'states': states, 'actions': actions, 'masks': masks, 'next_states':next_states,
                   'rewards': rewards, 'bad_masks': bad_masks}

    def load(self, file_name):
        raise NotImplemented