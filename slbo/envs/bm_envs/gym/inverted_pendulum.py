
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from slbo.utils.dataset import Dataset, gen_dtype
from lunzi.Logger import logger


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)

    def _step(self, a):
        # reward = 1.0
        reward = self._get_reward()
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        # notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        # done = not notdone
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reward(self):
        old_ob = self._get_obs()
        reward = -((old_ob[1]) ** 2)
        return reward

    def _get_obs(self):
        return np.concatenate([self.model.data.qpos, self.model.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent

    def mb_step(self, states, actions, next_states):
        # returns rewards and dones
        # forward rewards are calculated based on states, instead of next_states as in original SLBO envs
        if getattr(self, 'action_space', None):
            actions = np.clip(actions, self.action_space.low,
                              self.action_space.high)
        rewards = - self.cost_np_vec(states, actions, next_states)
        return rewards, np.zeros_like(rewards, dtype=np.bool)

    def cost_np_vec(self, obs, acts, next_obs):
        return ((obs[:, 1]) ** 2)

    def verify(self, n=2000, eps=1e-4):
        dataset = Dataset(gen_dtype(self, 'state action next_state reward done'), n)
        state = self.reset()
        for _ in range(n):
            action = self.action_space.sample()
            next_state, reward, done, _ = self.step(action)
            dataset.append((state, action, next_state, reward, done))

            state = next_state
            if done:
                state = self.reset()

        rewards_, dones_ = self.mb_step(dataset.state, dataset.action, dataset.next_state)
        diff = dataset.reward - rewards_
        l_inf = np.abs(diff).max()
        logger.info('rewarder difference: %.6f', l_inf)

        assert np.allclose(dones_, dataset.done)
        assert l_inf < eps
