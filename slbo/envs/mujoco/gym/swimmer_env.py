import numpy as np
from gym.envs.mujoco import swimmer
from stable_baselines import logger

from slbo.envs import BaseModelBasedEnv


# noinspection DuplicatedCode
class SwimmerEnv(swimmer.SwimmerEnv, BaseModelBasedEnv):
    def __init__(self, use_approximated_vel=True):
        self.use_approximated_vel = use_approximated_vel
        self.rescale_action = False

        if not self.use_approximated_vel:
            logger.warn('Modified Gym Env!')

        swimmer.SwimmerEnv.__init__(self)
        BaseModelBasedEnv.__init__(self)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # 5
            self.model.data.qvel.flat,  # 5
            self.get_body_com("torso").flat,  # 3
            self.get_body_comvel("torso").flat,  # 3
        ]).reshape(-1)

    def step(self, action):
        pre_pos = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        post_pos, height, ang = self.sim.data.qpos[0:3]
        if self.use_approximated_vel:
            fwd_reward = (post_pos - pre_pos) / self.dt
        else:
            fwd_reward = self.get_body_comvel('torso')[0]
        ctrl_reward = - 0.0001 * np.square(action).sum()
        reward = fwd_reward + ctrl_reward
        obs = self._get_obs()
        return obs, reward, False, {}

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        ctrl_reward = -  0.0001 * np.square(actions).sum(-1)
        fwd_reward = next_states[:, -3]
        reward = fwd_reward + ctrl_reward
        return reward, False
