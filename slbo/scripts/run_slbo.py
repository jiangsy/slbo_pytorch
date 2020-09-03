import os
import shutil
import time
from collections import deque
from operator import itemgetter

import numpy as np
import torch.utils.backcompat
import tqdm
from torch.utils.tensorboard import SummaryWriter

from slbo.algos import SLBO, PPO, TRPO
from slbo.configs.config import Config
from slbo.envs.wrapped_envs import make_vec_envs, make_vec_vritual_envs
from slbo.misc.ou_noise import OUNoise
from slbo.misc.utils import log_and_write, evaluate
from slbo.models import Actor, ActorCritic, Dynamics, VCritic, Normalizer
from slbo.storages.off_policy_buffer import OffPolicyBuffer
from slbo.storages.on_policy_buffer import OnPolicyBuffer

try:
    from slbo.misc import logger
except ImportError:
    from stable_baselines import logger


# noinspection DuplicatedCode
def main():
    config = Config('slbo_config.yaml')
    assert config.mf_algo == 'trpo'

    import datetime
    current_time = datetime.datetime.now().strftime('%b%d_%H%M%S')
    log_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'log')
    eval_log_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'log_eval')
    save_dir = os.path.join(config.proj_dir, config.result_dir, current_time, 'save')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    # save current version of code
    shutil.copytree(config.proj_dir, save_dir + '/code', ignore=shutil.ignore_patterns('result', 'data', 'ref'))

    device = torch.device('cuda' if config.use_cuda else 'cpu')

    # the normalization is done by the Normalizer
    real_envs = make_vec_envs(config.env.env_name, config.seed, config.env.num_envs, config.env.gamma, log_dir, device,
                              allow_early_resets=True, norm_reward=False, norm_obs=False, test=False)
    lo, hi = torch.tensor(real_envs.action_space.low, device=device), \
             torch.tensor(real_envs.action_space.high, device=device)

    state_dim = real_envs.observation_space.shape[0]
    action_space = real_envs.action_space
    action_dim = action_space.shape[0]

    normalizers = Normalizer(action_dim, state_dim)
    normalizers.to(device)

    dynamics = Dynamics(state_dim, action_dim, config.slbo.dynamics_hidden_dims, normalizer=normalizers)
    dynamics.to(device)

    noise = OUNoise(real_envs.action_space, config.ou_noise.theta, sigma=config.ou_noise.sigma)

    virt_envs = make_vec_vritual_envs(config.env.env_name, dynamics, config.seed, config.slbo.num_planning_envs,
                                      None, device, True)

    if config.mf_algo == 'ppo':
        actor_critic = ActorCritic(state_dim, action_space, actor_hidden_dims=[64, 64], critic_hidden_dims=[64, 64])
        actor_critic.to(device)

        agent = PPO(actor_critic, config.ppo_clip_param, config.ppo_num_grad_epochs, config.ppo_batch_size,
                    config.ppo_value_loss_coef,
                    config.ppo_entropy_loss_coef, lr=config.ppo_lr, max_grad_norm=config.ppo_max_grad_norm)
        noise_wrapped_actor = noise.wrap(actor_critic)
        mf_algo_config = config.ppo
        raise NotImplemented

    elif config.mf_algo == 'trpo':
        actor = Actor(state_dim, action_space, hidden_dims=config.trpo.actor_hidden_dims,
                      state_normalizer=normalizers.state_normalizer)
        # actor = MLPActor(state_dim, action_dim, config.trpo.actor_hidden_dims)
        critic = VCritic(state_dim, hidden_dims=config.trpo.critic_hidden_dims,
                         state_normalizer=normalizers.state_normalizer)
        actor.to(device)
        critic.to(device)
        agent = TRPO(actor, critic, entropy_coef=config.trpo.entropy_coef, verbose=config.verbose)
        noise_wrapped_actor = noise.wrap(actor)
        mf_algo_config = config.trpo

    # noinspection PyUnboundLocalVariable
    policy_buffer = \
        OnPolicyBuffer(mf_algo_config.num_env_steps, config.slbo.num_planning_envs, real_envs.observation_space.shape, action_space,
                       use_gae=mf_algo_config.use_gae, gamma=config.env.gamma, gae_lambda=mf_algo_config.gae_lambda,
                       use_proper_time_limits=mf_algo_config.use_proper_time_limits)
    policy_buffer.to(device)

    model = SLBO(dynamics, normalizers, config.slbo.batch_size, num_updates=config.slbo.num_model_updates,
                 num_rollout_steps=config.slbo.num_rollout_steps, lr=config.slbo.lr, l2_reg_coef=config.slbo.l2_reg_coef)

    model_buffer = OffPolicyBuffer(config.slbo.buffer_size, config.env.num_envs, state_dim, action_dim)
    model_buffer.to(device)

    if config.model_load_path is not None:
        actor_state_dict, critic_state_dict,  dynamics_state_dict, normalizers_state_dict \
            = itemgetter(*['actor', 'critic', 'dynamics', 'normalizers'])(torch.load(config.model_load_path))
        # noinspection PyUnboundLocalVariable
        actor.load_state_dict(actor_state_dict)
        dynamics.load_state_dict(dynamics_state_dict)
        # noinspection PyUnboundLocalVariable
        critic.load_state_dict(critic_state_dict)
        normalizers.load_state_dict(normalizers_state_dict)

    if config.buffer_load_path is not None:
        model_buffer.load(config.buffer_load_path)

    episode_rewards_real = deque(maxlen=10)
    episode_lengths_real = deque(maxlen=10)
    episode_rewards_virtual = deque(maxlen=10)
    episode_lengths_virtual = deque(maxlen=10)

    start = time.time()

    for epoch in range(config.slbo.num_epochs):
        logger.info('Epoch {}:'.format(epoch))

        if not config.slbo.use_prev_data:
            model_buffer.clear()

        # reset current_buffer and env
        cur_model_buffer = OffPolicyBuffer(config.slbo.num_env_steps, config.env.num_envs, state_dim, action_dim)
        cur_model_buffer.to(device)
        states = real_envs.reset()

        for step in range(config.slbo.num_env_steps):
            # noinspection PyUnboundLocalVariable
            unscaled_actions = noise_wrapped_actor.act(states)[0]
            actions = lo + (unscaled_actions + 1.) * 0.5 * (hi - lo)

            next_states, rewards, dones, infos = real_envs.step(actions)
            masks = torch.tensor([[0.0] if done else [1.0] for done in dones])
            bad_masks = torch.tensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            cur_model_buffer.insert(states, unscaled_actions, rewards, next_states, masks, bad_masks)

            states = next_states
            episode_rewards_real.extend([info['episode']['r'] for info in infos if 'episode' in info])
            episode_lengths_real.extend([info['episode']['l'] for info in infos if 'episode' in info])

            # mask out last collected tuples
            cur_model_buffer.bad_masks[-1, :, :] = 0.

        serial_env_steps = (epoch + 1) * config.slbo.num_env_steps
        total_env_steps = serial_env_steps * config.env.num_envs

        model_buffer.add_buffer(cur_model_buffer)

        if (epoch + 1) % config.slbo.log_interval == 0 and len(episode_rewards_real) > 0:
            log_info = [('serial_timesteps', serial_env_steps), ('total_timesteps', total_env_steps),
                        ('perf/ep_rew_real', np.mean(episode_rewards_real)),
                        ('perf/ep_len_real', np.mean(episode_lengths_real)),
                        ('time_elapsed', time.time() - start)
                        ]
            log_and_write(logger, writer, log_info, global_step=epoch * config.slbo.num_iters)

        if epoch == 0:
            samples = next(model_buffer.get_sequential_batch_generator(100, config.slbo.num_rollout_steps))
            states, next_states, masks, bad_masks = itemgetter(*['states', 'next_states', 'masks', 'bad_masks'])(samples)
            for i in range(config.slbo.num_rollout_steps - 1):
                assert torch.allclose(states[:, i + 1] * masks[:, i] * bad_masks[:, i],
                                      next_states[:, i] * masks[:, i] * bad_masks[:, i])

        if epoch == 50:
            # noinspection PyUnboundLocalVariable
            agent.entropy_coef = 0.

        normalizers.state_normalizer.update(cur_model_buffer.states.reshape([-1, state_dim]))
        # normalizers.action_normalizer.update(cur_model_buffer.actions)
        # FIXME: rule out incorrect value
        normalizers.diff_normalizer.update((cur_model_buffer.next_states - cur_model_buffer.states).
                                           reshape([-1, state_dim]))

        for i in range(config.slbo.num_iters):
            logger.log('Updating the model  - iter {:02d}'.format(i))
            losses = model.update(model_buffer)

            logger.log('Updating the policy - iter {:02d}'.format(i))
            for _ in tqdm.tqdm(range(config.slbo.num_policy_updates)):
                # collect data in the virtual env
                if config.slbo.start_strategy == 'buffer':
                    virt_envs.reset()  # set the need_reset flag to False
                    initial_states = next(model_buffer.get_batch_generator(config.slbo.num_planning_envs))['states']
                    for env_idx in range(config.slbo.num_planning_envs):
                        # FIXME: check set_state
                        virt_envs.env_method('set_state', initial_states[env_idx].cpu().numpy(), indices=env_idx)
                elif config.slbo.start_strategy == 'reset':
                    initial_states = virt_envs.reset()
                policy_buffer.states[0].copy_(initial_states)
                for step in range(config.trpo.num_env_steps):
                    with torch.no_grad():
                        actions, action_log_probs, dist_entropy, *_ = actor.act(policy_buffer.states[step])
                        values = critic(policy_buffer.states[step])

                    states, rewards, dones, infos = virt_envs.step(actions)

                    mask = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float32)
                    bad_mask = torch.tensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos],
                                            dtype=torch.float32)
                    policy_buffer.insert(states=states, actions=actions, action_log_probs=action_log_probs,
                                         values=values, rewards=rewards, masks=mask, bad_masks=bad_mask)

                    episode_rewards_virtual.extend([info['episode']['r'] for info in infos if 'episode' in info.keys()])
                    episode_lengths_virtual.extend([info['episode']['l'] for info in infos if 'episode' in info.keys()])

                with torch.no_grad():
                    next_value = critic(policy_buffer.states[-1])
                policy_buffer.compute_returns(next_value)
                losses.update(agent.update(policy_buffer))

            if (i + 1) % config.trpo.log_interval == 0 and len(episode_rewards_virtual) > 0:
                log_info = [('perf/ep_rew_virtual', np.mean(episode_rewards_virtual)),
                            ('perf/ep_len_virtual', np.mean(episode_lengths_virtual)),
                            ]
                for loss_name, loss_value in losses.items():
                    log_info.append(('loss/' + loss_name, loss_value))
                log_and_write(logger, writer, log_info, global_step=epoch * config.slbo.num_iters + i)

        if (epoch + 1) % config.save_freq == 0:
            torch.save({'actor': actor.state_dict(),
                        'critic': critic.state_dict(),
                        'dynamics': dynamics.state_dict(),
                        'normalizers': normalizers.state_dict()},
                       os.path.join(save_dir, 'actor_critic_dynamics_epoch{}.pt'.format(epoch)))
            logger.info('Saved model to {}'.format(os.path.join(save_dir,
                                                                'models_epoch{}.pt'.format(epoch))))

        if (epoch + 1) % config.eval_freq == 0:
            episode_rewards_real_eval, episode_lengths_real_eval = \
                evaluate(actor, config.env.env_name, config.seed, 10, None, device, False, False, False)
            log_info = [('perf/ep_rew_real_eval', np.mean(episode_rewards_real_eval)),
                        ('perf/ep_len_real_eval', np.mean(episode_lengths_real_eval)),
                        ]
            log_and_write(logger, writer, log_info, global_step=(epoch + 1) * config.slbo.num_iters)


if __name__ == '__main__':
    main()
