import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from slbo.envs import BaseModelBasedEnv


class RLLabHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):
    def __init__(self):
        self.rescale_action = True

        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), 'rllab_half_cheetah.xml'), 1)
        utils.EzPickle.__init__(self)

    def get_body_xmat(self, body_name):
        return self.sim.data.get_body_xmat(body_name)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,  # 9
            self.sim.data.qvel.flat,  # 9
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ])

    def step(self, action: np.ndarray):
        self.do_simulation(action, self.frame_skip)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        fwd_reward = self.get_body_comvel("torso")[0]
        ctrl_reward = - 0.05 * np.sum(np.square(action))
        reward = ctrl_reward + fwd_reward
        obs = self._get_obs()
        return obs, reward, False, {}

    def mb_step(self, states, actions, next_states):
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        ctrl_rewards = - 0.05 * np.sum(np.square(actions), axis=-1)
        fwd_rewards = next_states[..., 21]
        rewards = fwd_rewards + ctrl_rewards
        return rewards, np.zeros_like(fwd_rewards, dtype=np.bool)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(size=self.init_qpos.shape) * 0.01
        qvel = self.init_qvel + self.np_random.normal(size=self.init_qvel.shape) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
