import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from slbo.envs import BaseModelBasedEnv


class RLLabAntEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):
    def __init__(self):
        self.rescale_action = True

        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), 'rllab_ant.xml'), 1)
        utils.EzPickle.__init__(self)

    def get_body_xmat(self, body_name):
        return self.sim.data.get_body_xmat(body_name)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,  # 15
            self.sim.data.qvel.flat,  # 14
            self.get_body_xmat("torso").flat,  # 9
            self.get_body_com("torso").flat,  # 9 (should be 3?)
            self.get_body_comvel("torso").flat,  # 3
        ]).reshape(-1)

    def step(self, action: np.ndarray):
        self.do_simulation(action, self.frame_skip)
        comvel = self.get_body_comvel("torso")
        fwd_reward = comvel[0]
        scaling = (self.action_space.high - self.action_space.low) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.
        survive_reward = 0.05
        reward = fwd_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = not (np.isfinite(state).all() and 0.2 <= state[2] <= 1.0)
        obs = self._get_obs()
        return obs, float(reward), done, {}

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        comvel = next_states[..., -3:]
        fwd_reward = comvel[..., 0]
        scaling = (self.action_space.high - self.action_space.low) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(actions / scaling), axis=-1)
        contact_cost = 0.
        survive_reward = 0.05
        reward = fwd_reward - ctrl_cost - contact_cost + survive_reward
        notdone = np.all([next_states[..., 2] >= 0.2, next_states[..., 2] <= 1.0], axis=0)
        return reward, 1. - notdone

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(size=self.init_qpos.shape) * 0.01
        qvel = self.init_qvel + self.np_random.normal(size=self.init_qvel.shape) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
