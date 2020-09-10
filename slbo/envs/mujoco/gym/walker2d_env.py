from gym.envs.mujoco import walker2d
import numpy as np

from slbo.envs import BaseModelBasedEnv
from slbo.misc import logger


# noinspection DuplicatedCode
class Walker2DEnv(walker2d.Walker2dEnv, BaseModelBasedEnv):
    def __init__(self, use_approximated_vel=True):
        self.use_approximated_vel = use_approximated_vel
        self.rescale_action = False

        if not self.use_approximated_vel:
            logger.warn('Modified Gym Env!')

        walker2d.Walker2dEnv.__init__(self)
        BaseModelBasedEnv.__init__(self)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
            self.get_body_comvel("torso").flat
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
        ctrl_reward = - 1e-3 * np.square(action).sum()
        reward = fwd_reward + survive_reward + ctrl_reward
        done = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def mb_step(self, states, actions, next_states):
        if self.use_approximated_vel:
            fwd_rewards = (states[:, 0] - next_states[:, 0]) / self.dt
        else:
            fwd_rewards = next_states[:, 21]
        survive_rewards = 1.0
        ctrl_rewards = - 1e-3 * np.square(actions).sum(-1)
        rewards = fwd_rewards + survive_rewards + ctrl_rewards
        dones = ~((0.8 < next_states[:, 1] < 2.0) &
                  (-1.0 < next_states[:, 2] < 1.0))
        return rewards, dones
