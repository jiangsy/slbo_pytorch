import os
from typing import Optional

import gym
import torch
from stable_baselines import bench
from stable_baselines.common.vec_env import VecEnvWrapper
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

from slbo.envs.mujoco.mujoco_envs import make_mujoco_env
from slbo.envs.virtual_env import VirtualEnv
from slbo.models.dynamics import Dynamics


def make_env(env_id, seed, rank, log_dir, allow_early_resets, test=True):
    def _thunk():
        if test:
            env = gym.make(env_id)
        else:
            env = make_mujoco_env(env_id)
        env.seed(seed + rank)
        try:
            if env._max_episode_steps > 0:
                env = TimeLimitMask(env)
        except AttributeError:
            pass

        log_dir_ = os.path.join(log_dir, str(rank)) if log_dir is not None else log_dir
        env = bench.Monitor(env, log_dir_, allow_early_resets=allow_early_resets)

        return env

    return _thunk


def make_vec_envs(env_name: str,
                  seed: int,
                  num_envs: int,
                  gamma: float,
                  log_dir: str,
                  device: torch.device,
                  allow_early_resets: bool,
                  norm_reward=True,
                  norm_obs=True,
                  test=False,
                  ):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, test)
        for i in range(num_envs)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, norm_reward=False, norm_obs=norm_obs)
        else:
            envs = VecNormalize(envs, gamma=gamma, norm_reward=norm_reward, norm_obs=norm_obs)

    envs = VecPyTorch(envs, device)

    return envs


def make_virtual_env(env_name, dynamics: Dynamics, seed, rank, allow_early_resets):
    def _thunk():
        virt_env = VirtualEnv(dynamics, make_mujoco_env(env_name), seed + rank)
        env = bench.Monitor(virt_env, None, allow_early_resets=allow_early_resets)
        return env

    return _thunk


def make_vec_vritual_envs(env_name: str,
                          dynamics: Dynamics,
                          seed: int,
                          num_envs: int,
                          gamma: Optional[float],
                          device: torch.device,
                          allow_early_resets: bool,
                          norm_reward=False,
                          norm_obs=False,
                          ):
    envs = [
        make_virtual_env(env_name, dynamics, seed, i, allow_early_resets)
        for i in range(num_envs)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1 and (norm_reward or norm_obs):
        if gamma is None:
            envs = VecNormalize(envs, norm_reward=False, norm_obs=norm_obs)
        else:
            envs = VecNormalize(envs, gamma=gamma, norm_reward=norm_reward, norm_obs=norm_obs)

    envs = VecPyTorch(envs, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def mb_step_wait(self):
        pass


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None


