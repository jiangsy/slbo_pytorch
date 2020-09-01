import gym
import numpy as np
import torch

from slbo.envs import BaseModelBasedEnv
from slbo.models.dynamics import Dynamics


class VirtualEnv(gym.Env):
    def __init__(self, dynamics: Dynamics, env: BaseModelBasedEnv, seed):
        super().__init__()
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.dynamics = dynamics
        self.env = env
        self.env.seed(seed)

        self.state = np.zeros([self.observation_space.shape[0]], dtype=np.float32)

    def step(self, action: np.ndarray):
        device = next(self.dynamics.parameters()).device

        states = self.state.reshape([1, self.state_dim])
        actions = action.reshape([1, self.action_dim])
        with torch.no_grad():
            next_states = self.dynamics(torch.tensor(states, device=device, dtype=torch.float32),
                                        torch.tensor(actions, device=device, dtype=torch.float32)).cpu().numpy()
            reward, done = self.env.mb_step(states, actions, next_states)
            reward, done = reward[0], done[0]
        self.state = next_states[0]
        return self.state.copy(), reward.copy(), done.copy(), {}

    def reset(self) -> np.ndarray:
        self.state = self.env.reset()
        return self.state.copy()

    def set_state(self, state: np.ndarray):
        self.state = state.copy()

    def render(self, mode='human'):
        raise NotImplemented



