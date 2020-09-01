import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from slbo.envs import BaseModelBasedEnv


class RLLabHopperEnv(mujoco_env.MujocoEnv, utils.EzPickle, BaseModelBasedEnv):
    def __init__(self):
        self.rescale_action = True

        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.path.dirname(__file__), 'rllab_hopper.xml'), 1)
        utils.EzPickle.__init__(self)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,  # 6
            self.sim.data.qvel.flat,  # 6
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso"),  # 3
        ])

    def step(self, action: np.ndarray):
        self.do_simulation(action, self.frame_skip)
        scaling = 0.5 * (self.action_space.high - self.action_space.low)
        vel = self.get_body_comvel("torso")[0]
        alive_bonus = 1.0
        reward = vel + alive_bonus - 0.005 * np.sum(np.square(action / scaling))
        # FIXME
        state = self.state_vector()
        done = not (np.isfinite(state).all() and
                   (np.abs(state[3:]) < 100).all() and (state[0] > .7) and
                   (abs(state[2]) < .2))
        obs = self._get_obs()
        return obs, reward, done, {}

    def mb_step(self, states, actions, next_states):
        scaling = (self.action_space.high - self.action_space.low) * 0.5
        vel = next_states[:, -3]
        alive_bonus = 1.0
        reward = vel + alive_bonus - 0.005 * np.sum(np.square(actions / scaling), axis=-1)

        done = ~((next_states[:, 3:12] < 100).all(axis=-1) &
                 (next_states[:, 0] > 0.7) &
                 (np.abs(next_states[:, 2]) < 0.2))
        return reward, done

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(size=self.init_qpos.shape) * 0.01
        qvel = self.init_qvel + self.np_random.normal(size=self.init_qvel.shape) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()