from typing import List

import torch
from torch.utils.tensorboard import SummaryWriter

from slbo.envs.wrapped_envs import make_vec_envs, get_vec_normalize


def log_and_write(logger, writer: SummaryWriter, log_infos: List, global_step: int):
    for idx, (name, value) in enumerate(log_infos):
        if logger is not None:
            logger.logkv('{}.'.format(idx) + name.split('/')[-1], value)
        if writer is not None and name.find('/') > -1:
            writer.add_scalar(name, value, global_step=global_step)
    if logger is not None:
        logger.dumpkvs()


def collect_traj(actor, envs, buffer, total_step):
    episode_rewards = []
    episode_lengths = []

    step = 0
    while step < total_step:
        states = envs.reset()
        dones = False
        traj = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'masks': []}
        while not dones:
            with torch.no_grad():
                actions, *_ = actor(states, deterministic=False, reparameterize=False)

            new_states, rewards, dones, infos = envs.step_index(actions)
            mask = torch.tensor([[0.0] if done_ else [1.0] for done_ in dones], dtype=torch.float32)

            traj['states'].append(states)
            traj['actions'].append(actions)
            traj['next_states'].append(new_states)
            traj['rewards'].append(rewards)
            traj['masks'].append(mask)

            states = new_states

            for info_ in infos:
                if 'episode' in info_.keys():
                    episode_rewards.append(info_['episode']['r'])
                    episode_lengths.append(info_['episode']['l'])

        traj_len = len(traj['actions'])
        step += traj_len
        buffer.add_traj(traj)

    return episode_rewards, episode_lengths


def evaluate(actor, env_name, seed, num_episode, eval_log_dir,
             device, max_episode_steps=1000, norm_reward=False, norm_obs=True, obs_rms=None, test=True):
    eval_envs = make_vec_envs(env_name, seed + 1, 1, None, eval_log_dir, device, True,
                              max_episode_steps, norm_reward, norm_obs, test)

    vec_norm = get_vec_normalize(eval_envs)
    if vec_norm is not None and norm_obs:
        assert obs_rms is not None
        vec_norm.training = False
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []
    eval_episode_lengths = []

    obs = eval_envs.reset()

    while len(eval_episode_rewards) < num_episode:
        with torch.no_grad():
            action, *_ = actor.act(obs, deterministic=True)

        obs, _, done, infos = eval_envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
                eval_episode_lengths.append(info['episode']['l'])

    eval_envs.close()

    return eval_episode_rewards, eval_episode_lengths
