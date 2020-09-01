from gym.envs.mujoco import ant
import numpy as np
from stable_baselines import logger

from slbo.envs import BaseModelBasedEnv


# noinspection DuplicatedCode
class AntEnv(ant.AntEnv, BaseModelBasedEnv):
    def __init__(self, use_approximated_vel=True):
        logger.warn('Modified Gym Envs!')
        self.rescale_action = False
        self.use_approximated_vel = use_approximated_vel

        ant.AntEnv.__init__(self)
        BaseModelBasedEnv.__init__(self)

    def get_body_xmat(self, body_name):
        return self.sim.data.get_body_xmat(body_name)

    def get_body_comvel(self, body_name):
        return self.sim.data.get_body_xvelp(body_name)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,  # 15
            self.sim.data.qvel.flat,  # 14
            self.get_body_xmat("torso").flat,  # 9
            self.get_body_com("torso"),  # 9
            self.get_body_comvel("torso"),  # 3
        ]).reshape(-1)

    def step(self, action):
        pre_pos = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        post_pos = self.sim.data.qpos[0]
        if self.use_approximated_vel:
            fwd_reward = (post_pos - pre_pos) / self.dt
        else:
            fwd_reward = self.get_body_comvel('torso')[0]
        ctrl_reward = - .5 * np.square(action).sum()
        # make sure the reward can be recovered from state and action completely
        contact_reward = - 0.
        survive_reward = 1.0
        reward = fwd_reward + ctrl_reward + contact_reward + survive_reward
        state = self.state_vector()
        done = not(np.isfinite(state).all() and 0.2 <= state[2] <= 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        if self.use_approximated_vel:
            reward_forward = (next_states[:, 0] - states[:, 0]) / self.dt
        else:
            reward_forward = next_states[..., -3]

        ctrl_cost = .5 * np.square(actions).sum(-1)
        contact_cost = 0.
        survive_reward = 1.0
        reward = reward_forward - ctrl_cost - contact_cost + survive_reward
        notdone = np.all(0.2 <= next_states[..., 2] <= 1.0, axis=0)
        return reward, 1. - notdone

