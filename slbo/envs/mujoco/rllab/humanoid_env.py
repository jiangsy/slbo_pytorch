import os

import gym.utils as utils
import numpy as np
from gym.envs.mujoco import mujoco_env

from slbo.envs import BaseModelBasedEnv


class RLLabSimpleHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):
    def __init__(self):
        self.rescale_action = True

        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), 'rllab_simple_humanoid.xml'), 1)
        utils.EzPickle.__init__(self)

    def get_body_xmat(self, body_name):
        return self.sim.data.get_body_xmat(body_name)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([
            data.qpos.flat,  # 17
            data.qvel.flat,  # 16
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ])

    def step(self, actions: np.ndarray):
        alive_bonus = 0.2
        comvel = self.get_body_comvel("torso")
        lin_vel_reward = comvel[0]
        scaling = 0.5 * (self.action_space.high - self.action_space.low)
        ctrl_cost = 5e-4 * np.sum(np.square(actions / scaling))
        impact_cost = 0.
        vel_deviation_cost = 5e-3 * np.sum(np.square(comvel[1:]))
        reward = lin_vel_reward + alive_bonus - ctrl_cost - impact_cost - vel_deviation_cost
        done = not (0.8 <= self.sim.data.qpos.flat[2] <= 2.0)
        next_obs = self._get_obs()
        return next_obs, reward, done, {}

    def mb_step(self, states, actions, next_states):
        scaling = 0.5 * (self.action_space.high - self.action_space.low)

        alive_bonus = 0.2
        lin_vel_reward = next_states[:, 36]
        ctrl_cost = 5.e-4 * np.square(actions / scaling).sum(axis=1)
        impact_cost = 0.
        vel_deviation_cost = 5.e-3 * np.square(next_states[:, 37:39]).sum(axis=1)
        reward = lin_vel_reward + alive_bonus - ctrl_cost - impact_cost - vel_deviation_cost

        dones = not (0.8 <= next_states[:, 2] <= 2.0)
        return reward, dones

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(size=self.init_qpos.shape) * 0.01
        qvel = self.init_qvel + self.np_random.normal(size=self.init_qvel.shape) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
