import os

import gym.utils as utils
import numpy as np
from gym.envs.mujoco import mujoco_env

from slbo.envs import BaseModelBasedEnv


class RLLabWalker2dEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):
    def __init__(self):
        self.rescale_action = True

        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), 'rllab_walker2d.xml'), 1)
        utils.EzPickle.__init__(self)

    def get_body_xmat(self, body_name):
        return self.sim.data.get_body_xmat(body_name)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
            self.get_body_comvel("torso").flat
        ])

    def step(self, action: np.ndarray):
        self.do_simulation(action, self.frame_skip)
        fwd_reward = self.get_body_comvel("torso")[0]
        scaling = 0.5 * (self.action_space.high - self.action_space.low)
        ctrl_cost = 1e-3 * np.sum(np.square(action / scaling))
        alive_bonus = 1.
        reward = fwd_reward - ctrl_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = not (0.8 < qpos[0] < 2.0 and -1.0 < qpos[2] < 1.0)
        obs = self._get_obs()
        return obs, reward, done, {}

    def mb_step(self, states, actions, next_states):
        scaling = 0.5 * (self.action_space.high - self.action_space.low)
        reward_ctrl = -0.001 * np.sum(np.square(actions / scaling), axis=-1)
        reward_fwd = next_states[:, 21]
        alive_bonus = 1.
        rewards = reward_ctrl + reward_fwd + alive_bonus
        dones = not ((0.8 < next_states[:, 0] < 2.0) and (-1.0 < next_states[:, 2] < 1.0))
        return rewards, dones

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(size=self.init_qpos.shape) * 0.01
        qvel = self.init_qvel + self.np_random.normal(size=self.init_qvel.shape) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
