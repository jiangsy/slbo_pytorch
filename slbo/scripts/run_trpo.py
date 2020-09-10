import shutil
import time
from collections import deque

import numpy as np
import torch
import torch.backends.cudnn

from torch.utils.tensorboard import SummaryWriter
import os

from slbo.algos.mfrl.trpo import TRPO
from slbo.configs.config import Config
from slbo.envs.wrapped_envs import make_vec_envs, get_vec_normalize
from slbo.models import Actor, VCritic
from slbo.misc.utils import evaluate, log_and_write
from slbo.storages.on_policy_buffer import OnPolicyBuffer
from slbo.misc import logger


# noinspection DuplicatedCode
def main():
    logger.info('Test script for TRPO')
    config, hparam_dict = Config('trpo_config.yaml')

    torch.manual_seed(config.seed)
    # noinspection PyUnresolvedReferences
    torch.cuda.manual_seed_all(config.seed)

    if config.use_cuda and torch.cuda.is_available() and config.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    import datetime
    current_time = datetime.datetime.now().strftime('%b%d_%H%M%S')
    log_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'log')
    eval_log_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'log_eval')
    save_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'save')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_hparams(hparam_dict, {})

    # save current version of code
    shutil.copytree(config.proj_dir, save_dir + '/code', ignore=shutil.ignore_patterns('result', 'data', 'ref'))

    torch.set_num_threads(1)
    device = torch.device('cuda' if config.use_cuda else 'cpu')

    envs = make_vec_envs(config.env.env_name, config.seed, config.env.num_envs, config.env.gamma, log_dir, device,
                         allow_early_resets=False, norm_reward=True, norm_obs=True, test=True)

    state_dim = envs.observation_space.shape[0]
    action_space = envs.action_space
    action_dim = action_space.shape[0]

    actor = Actor(state_dim, action_space, hidden_dims=config.trpo.actor_hidden_dims,
                  state_normalizer=None)
    critic = VCritic(state_dim, hidden_dims=config.trpo.critic_hidden_dims, state_normalizer=None)
    actor.to(device)
    critic.to(device)

    agent = TRPO(actor, critic,)

    on_policy_buffer = \
        OnPolicyBuffer(config.trpo.num_env_steps, config.env.num_envs, envs.observation_space.shape, envs.action_space,
                       use_gae=config.trpo.use_gae, gamma=config.env.gamma, gae_lambda=config.trpo.gae_lambda,
                       use_proper_time_limits=config.trpo.use_proper_time_limits, )

    state = envs.reset()
    # noinspection PyUnresolvedReferences
    on_policy_buffer.states[0].copy_(state)
    on_policy_buffer.to(device)

    episode_rewards = deque(maxlen=10)
    episode_lengths = deque(maxlen=10)

    start = time.time()
    num_updates = config.trpo.total_env_steps // config.trpo.num_env_steps // config.env.num_envs

    for j in range(num_updates):

        for step in range(config.trpo.num_env_steps):
            with torch.no_grad():
                action, action_log_prob, dist_entropy, *_ = actor.act(on_policy_buffer.states[step])
                value = critic(on_policy_buffer.states[step])

            state, reward, done, info = envs.step(action)

            for info_ in info:
                if 'episode' in info_.keys():
                    episode_rewards.append(info_['episode']['r'])
                    episode_lengths.append(info_['episode']['l'])

            mask = torch.tensor([[0.0] if done_ else [1.0] for done_ in done], dtype=torch.float32)
            bad_mask = torch.tensor([[0.0] if 'bad_transition' in info_.keys() else [1.0] for info_ in info],
                                    dtype=torch.float32)
            on_policy_buffer.insert(states=state, actions=action, action_log_probs=action_log_prob,
                                    values=value, rewards=reward, masks=mask, bad_masks=bad_mask)

        with torch.no_grad():
            next_value = critic(on_policy_buffer.states[-1])

        on_policy_buffer.compute_returns(next_value)
        losses = agent.update(on_policy_buffer)
        on_policy_buffer.after_update()

        if j % config.save_interval == 0 or j == num_updates - 1:
            save_path = os.path.join(save_dir, config.mf_algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            logger.info('Model saved.')
            torch.save([actor.state_dict(), critic.state_dict(),
                        getattr(get_vec_normalize(envs), 'obs_rms', None)],
                       os.path.join(save_path, config.env.env_name + ".pt"))

        serial_timsteps = (j + 1) * config.trpo.num_env_steps
        total_num_steps = config.env.num_envs * serial_timsteps
        end = time.time()

        fps = int(total_num_steps / (end - start))

        if j % config.log_interval == 0 and len(episode_rewards) > 0:
            log_info = [('serial_timesteps', serial_timsteps), ('total_timesteps', total_num_steps),
                        ('ep_rew_mean', np.mean(episode_rewards)), ('ep_len_mean', np.mean(episode_lengths)),
                        ('fps', fps), ('time_elapsed', end - start)]

            for loss_name, loss_value in losses.items():
                log_info.append((loss_name, loss_value))
            log_and_write(logger, writer, log_info, global_step=j)

        if (config.eval_interval is not None and len(episode_rewards) > 0
                and j % config.eval_interval == 0):
            obs_rms = get_vec_normalize(envs).obs_rms
            eval_episode_rewards, eval_episode_lengths = \
                evaluate(actor, config.env.env_name, config.seed,
                         num_episode=10, eval_log_dir=None, device=device, norm_reward=True, norm_obs=True,
                         obs_rms=obs_rms, test=True)

            logger.info('Evaluation:')
            log_and_write(logger, writer, [('eval_ep_rew_mean', np.mean(eval_episode_rewards)),
                                           ('eval_ep_rew_min', np.min(eval_episode_rewards)),
                                           ('eval_ep_rew_max', np.max(eval_episode_rewards))], global_step=j)

    envs.close()


if __name__ == "__main__":
    main()
