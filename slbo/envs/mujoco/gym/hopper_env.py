from gym.envs.mujoco import hopper
import numpy as np

from slbo.envs import BaseModelBasedEnv
from slbo.misc import logger


# noinspection DuplicatedCode
class HopperEnv(hopper.HopperEnv, BaseModelBasedEnv):
    def __init__(self, use_approximated_vel=True):
        self.use_approximated_vel = use_approximated_vel
        self.rescale_action = False

        if not self.use_approximated_vel:
            logger.warn('Modified Gym Env!')
        hopper.HopperEnv.__init__(self)
        BaseModelBasedEnv.__init__(self)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,  # 6
            self.sim.data.qvel.flat,  # 6
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ])

    def step(self, action):
        pre_pos = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        post_pos, height, ang = self.sim.data.qpos[0:3]
        if self.use_approximated_vel:
            fwd_reward = (post_pos - pre_pos) / self.dt
        else:
            fwd_reward = self.get_body_comvel('torso')[0]
        survive_reward = 1.0
        ctrl_reward = -1e-3 * np.square(action).sum()
        reward = fwd_reward + survive_reward + ctrl_reward
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def mb_step(self, states, actions, next_states):
        if self.use_approximated_vel:
            fwd_reward = (next_states[:, 0] - states[:, 0]) / self.dt
        else:
            fwd_reward = next_states[:, -3]

        survive_reward = 1.0
        ctrl_reward = -1e-3 * np.square(actions).sum(-1)

        reward = fwd_reward + survive_reward + ctrl_reward

        done = ~((next_states[:, 2:12] < 100).all(axis=-1) &
                 (next_states[:, 1] > 0.7) &
                 (np.abs(next_states[:, 2]) < 0.2))
        return reward, done


