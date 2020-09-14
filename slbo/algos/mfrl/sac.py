import torch
from torch import nn as nn

from slbo.models.utils import soft_update
from slbo.models.q_critic import QCritic


class SAC:
    def __init__(
            self,
            actor,
            q_critic1: QCritic,
            q_critic2: QCritic,
            target_q_critic1: QCritic,
            target_q_critic2: QCritic,
            batch_size,
            num_grad_steps,
            gamma=0.99,
            reward_scale=1.0,
            actor_lr=1e-3,
            critic_lr=1e-3,
            soft_target_tau=1e-2,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            alpha=1.0,
    ):
        super(SAC).__init__()
        self.actor = actor
        self.q_critic1 = q_critic1
        self.q_critic2 = q_critic2
        self.target_q_critic1 = target_q_critic1
        self.target_q_critic2 = target_q_critic2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.batch_size = batch_size
        self.num_grad_steps = num_grad_steps

        self.automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = torch.tensor(target_entropy)
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=actor_lr)

        self._alpha = torch.tensor(alpha)

        self.qf_criterion = nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.qf1_optimizer = torch.optim.Adam(self.q_critic1.parameters(), lr=critic_lr)
        self.qf2_optimizer = torch.optim.Adam(self.q_critic2.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.reward_scale = reward_scale
        self.num_updates = 0
        self._need_to_update_eval_statistics = True

    def update(self, policy_buffer) -> dict:

        data_generator = policy_buffer.get_batch_generator(self.batch_size)

        self.actor.train()
        self.q_critic1.train()
        self.q_critic2.train()
        self.target_q_critic1.train()
        self.target_q_critic2.train()

        policy_loss_epoch = 0.
        qf1_loss_epoch = 0.
        qf2_loss_epoch = 0.
        alpha_loss_epoch = 0.

        for _ in range(self.num_grad_steps):

            try:
                batch = next(data_generator)
            except StopIteration:
                data_generator = policy_buffer.get_batch_generator(self.batch_size)
                batch = next(data_generator)

            states, actions, rewards, masks, next_states = \
                batch['states'], batch['actions'], batch['rewards'], batch['masks'], batch['next_states']

            new_actions, log_probs, *_ = self.actor(states, return_log_prob=True)

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha.to(log_probs.device) * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = torch.tensor([0.])
                alpha = self._alpha

            alpha = alpha.to(log_probs.device)

            q_new_actions = torch.min(self.q_critic1(states, new_actions),
                                      self.q_critic2(states, new_actions))
            policy_loss = (alpha * log_probs - q_new_actions).mean()

            q1_pred = self.q_critic1(states, actions)
            q2_pred = self.q_critic2(states, actions)

            new_next_actions, new_next_log_probs, *_ = self.actor(next_states, return_log_prob=True)

            target_q_values = torch.min(self.target_q_critic1(next_states, new_next_actions),
                                        self.target_q_critic2(next_states, new_next_actions)) - alpha * new_next_log_probs

            q_target = self.reward_scale * rewards + masks * self.gamma * target_q_values
            qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
            qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

            if self.num_updates % self.target_update_period == 0:
                soft_update(self.q_critic1, self.target_q_critic1, self.soft_target_tau)
                soft_update(self.q_critic2, self.target_q_critic2, self.soft_target_tau)

            self.num_updates += 1

            policy_loss_epoch += policy_loss.item()
            qf1_loss_epoch += qf1_loss.item()
            qf2_loss_epoch += qf2_loss.item()
            alpha_loss_epoch += alpha_loss.item()

        policy_loss_epoch /= self.num_grad_steps
        qf1_loss_epoch /= self.num_grad_steps
        qf2_loss_epoch /= self.num_grad_steps
        alpha_loss_epoch /= self.num_grad_steps

        return {'policy_loss': policy_loss_epoch, 'qf1_loss': qf1_loss_epoch,
                'qf2_loss': qf2_loss_epoch, 'alpha_loss': alpha_loss_epoch}

