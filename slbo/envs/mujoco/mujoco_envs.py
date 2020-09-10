import numpy as np

from slbo.envs.mujoco.gym.ant_env import AntEnv
from slbo.envs.mujoco.gym.half_cheetah_env import HalfCheetahEnv
from slbo.envs.mujoco.gym.hopper_env import HopperEnv
from slbo.envs.mujoco.gym.swimmer_env import SwimmerEnv
from slbo.envs.mujoco.gym.walker2d_env import Walker2DEnv
from slbo.envs.mujoco.rllab.ant_env import RLLabAntEnv
from slbo.envs.mujoco.rllab.half_cheetah_env import RLLabHalfCheetahEnv
from slbo.envs.mujoco.rllab.hopper_env import RLLabHopperEnv
from slbo.envs.mujoco.rllab.humanoid_env import RLLabSimpleHumanoidEnv
from slbo.envs.mujoco.rllab.swimmer_env import RLLabSwimmerEnv
from slbo.envs.mujoco.rllab.walker2d_env import RLLabWalker2dEnv


def make_mujoco_env(env_name: str):
    envs = {
        'HalfCheetah-v2': HalfCheetahEnv,
        'Walker2D-v2': Walker2DEnv,
        'Ant-v2': AntEnv,
        'Hopper-v2': HopperEnv,
        'Swimmer-v2': SwimmerEnv,
        'RLLabHalfCheetah-v2': RLLabHalfCheetahEnv,
        'RLLabWalker2D-v2': RLLabWalker2dEnv,
        'RLLabAnt-v2': RLLabAntEnv,
        'RLLabHopper-v2': RLLabHopperEnv,
        'RLLabSwimmer-v2': RLLabSwimmerEnv,
        'RLLabHumanoid-v2': RLLabSimpleHumanoidEnv
    }
    env = envs[env_name]()
    if not hasattr(env, 'reward_range'):
        env.reward_range = (-np.inf, np.inf)
    if not hasattr(env, 'metadata'):
        env.metadata = {}
    env.seed(np.random.randint(2 ** 60))
    return env
