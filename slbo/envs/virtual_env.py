from __future__ import annotations

import gym
import numpy as np
import torch
from typing import TYPE_CHECKING

from slbo.thirdparty.base_vec_env import VecEnv
if TYPE_CHECKING:
    from slbo.models.dynamics import Dynamics
    from slbo.envs import BaseModelBasedEnv


class VirtualEnv(gym.Env):
    def __init__(self, dynamics: Dynamics, env: BaseModelBasedEnv, seed):
        super().__init__()
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.dynamics = dynamics
        self.device = next(self.dynamics.parameters()).device
        self.env = env
        self.env.seed(seed)

        self.state = np.zeros([self.observation_space.shape[0]], dtype=np.float32)

    def _rescale_action(self, action):
        lo, hi = self.action_space.low, self.action_space.high
        return lo + (action + 1.) * 0.5 * (hi - lo)

    def step_await(self, action: np.ndarray):
        states = self.state.reshape([1, self.state_dim])
        actions = action.reshape([1, self.action_dim])
        rescaled_actions = self._rescale_action(action).reshape([1, self.action_dim])
        with torch.no_grad():
            next_states = self.dynamics(torch.tensor(states, device=self.device, dtype=torch.float32),
                                        torch.tensor(actions, device=self.device, dtype=torch.float32)).cpu().numpy()
            rewards, dones = self.env.mb_step(states, rescaled_actions, next_states)
            reward, done = rewards[0], dones[0]
        self.state = next_states[0]
        return self.state.copy(), reward.copy(), done.copy(), {}

    def reset(self) -> np.ndarray:
        self.state = self.env.reset()
        return self.state.copy()

    def set_state(self, state: np.ndarray):
        self.state = state.copy()

    def render(self, mode='human'):
        raise NotImplemented


class VecVirtualEnv(VecEnv):
    def __init__(self, dynamics: Dynamics, env: BaseModelBasedEnv, num_envs, seed, max_episode_steps=1000,
                 auto_reset=True):
        super(VecEnv, self).__init__()
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.auto_reset = auto_reset

        self.dynamics = dynamics
        self.device = next(self.dynamics.parameters()).device
        self.env = env
        self.env.seed(seed)

        self.elapsed_steps = np.zeros([self.num_envs], dtype=np.int32)
        self.episode_rewards = np.zeros([self.num_envs])

        self.states = np.zeros([self.num_envs, self.observation_space.shape[0]], dtype=np.float32)

    def _rescale_action(self, actions: np.array):
        lo, hi = self.action_space.low, self.action_space.high
        return lo + (actions + 1.) * 0.5 * (hi - lo)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        rescaled_actions = self._rescale_action(self.actions)
        self.elapsed_steps += 1
        with torch.no_grad():
            next_states = self.dynamics(torch.tensor(self.states, device=self.device, dtype=torch.float32),
                                        torch.tensor(self.actions, device=self.device, dtype=torch.float32)).cpu().numpy()
        rewards, dones = self.env.mb_step(self.states, rescaled_actions, next_states)
        self.episode_rewards += rewards
        self.states = next_states.copy()
        timeouts = self.elapsed_steps == self.max_episode_steps
        dones |= timeouts
        info_dicts = [{} for _ in range(self.num_envs)]
        for i, (done, timeout) in enumerate(zip(dones, timeouts)):
            if done:
                info = {'episode': {'r': self.episode_rewards[i], 'l': self.elapsed_steps[i]}}
                if timeout:
                    info.update({'TimeLimit.truncated': True})
                info_dicts[i] = info
            else:
                info_dicts[i] = {}
        if self.auto_reset:
            self.reset(np.argwhere(dones).squeeze(axis=-1))
        return self.states.copy(), rewards.copy(), dones.copy(), info_dicts

    # if indices = None, every env will be reset
    def reset(self, indices=None) -> np.ndarray:
        # have to distinguish [] and None
        indices = np.arange(self.num_envs) if indices is None else indices
        if np.size(indices) == 0:
            return np.array([])
        states = np.array([self.env.reset() for _ in indices])
        self.states[indices] = states
        self.elapsed_steps[indices] = 0
        self.episode_rewards[indices] = 0.
        return states.copy()

    # if indices = None, every env will be set
    def set_state(self, states: np.ndarray, indices=None):
        indices = indices or np.arange(self.num_envs)
        assert states.ndim == 2 and states.shape[0] == indices.shape[0]
        self.states[indices] = states.copy()
        # set_state should reset reward and length
        self.elapsed_steps[indices] = 0
        self.episode_rewards[indices] = 0.

    def close(self):
        pass

    def seed(self, seed):
        return self.env.seed(seed)

    def render(self, mode='human'):
        raise NotImplemented

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplemented

    def get_attr(self, attr_name, indices=None):
        raise NotImplemented

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplemented

