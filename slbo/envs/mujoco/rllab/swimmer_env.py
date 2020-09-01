import os

import gym.utils as utils
import numpy as np
from gym.envs.mujoco import mujoco_env

from slbo.envs import BaseModelBasedEnv


class RLLabSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):
    def __init__(self):
        self.rescale_action = True

        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), 'rllab_swimmer.xml'), 50)
        utils.EzPickle.__init__(self)

    def get_body_xmat(self, body_name):
        return self.sim.data.get_body_xmat(body_name)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,  # 5
            self.sim.data.qvel.flat,  # 5
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ]).reshape(-1)

    def step(self, action: np.ndarray):
        self.do_simulation(action, self.frame_skip)
        scaling = 0.5 * (self.action_space.high - self.action_space.low)
        ctrl_cost = 0.005 * np.sum(np.square(action / scaling))
        fwd_reward = self.get_body_comvel("torso")[0]
        reward = fwd_reward - ctrl_cost
        obs = self._get_obs()
        return obs, reward, False, {}

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        scaling = 0.5 * (self.action_space.high - self.action_space.low)
        ctrl_cost = 0.005 * np.sum(np.square(actions / scaling), axis=-1)
        fwd_reward = next_states[:, -3]
        reward = fwd_reward - ctrl_cost
        return reward, np.zeros_like(reward, dtype=np.bool)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(size=self.init_qpos.shape) * 0.01
        qvel = self.init_qvel + self.np_random.normal(size=self.init_qvel.shape) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
