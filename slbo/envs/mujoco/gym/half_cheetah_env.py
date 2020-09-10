from gym.envs.mujoco import half_cheetah
import numpy as np

from slbo.envs import BaseModelBasedEnv
from slbo.misc import logger


# noinspection DuplicatedCode
class HalfCheetahEnv(half_cheetah.HalfCheetahEnv, BaseModelBasedEnv):
    def __init__(self, use_approximated_vel=True):
        self.use_approximated_vel = use_approximated_vel
        self.rescale_action = False
        if not self.use_approximated_vel:
            logger.warn('Modified Gym Env!')

        half_cheetah.HalfCheetahEnv.__init__(self)
        BaseModelBasedEnv.__init__(self)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # 9
            self.model.data.qvel.flat,  # 9
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ])

    def step(self, action: np.ndarray):
        pre_pos = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        post_pos = self.sim.data.qpos[0]
        if self.use_approximated_vel:
            fwd_reward = (post_pos - pre_pos) / self.dt
        else:
            fwd_reward = self.get_body_comvel('torso')[0]
        ctrl_reward = - 0.1 * np.square(action).sum()
        reward = ctrl_reward + fwd_reward
        obs = self._get_obs()
        return obs, reward, False, {}

    def mb_step(self, states, actions, next_states):
        ctrl_rewards = - 0.1 * np.square(actions).sum(-1)
        if self.use_approximated_vel:
            fwd_rewards = (next_states[:, 0] - states[:, 0]) / self.dt
        else:
            fwd_rewards = next_states[:, 21]
        rewards = fwd_rewards + ctrl_rewards
        return rewards, np.zeros_like(rewards, dtype=np.bool)
