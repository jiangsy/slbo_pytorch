import torch
import torch.nn as nn
import torch.optim as optim

from slbo.models.actor_critic import ActorCritic


class PPO:
    def __init__(self, actor_critic: ActorCritic, clip_param: float, num_grad_updates: int, batch_size: int,
                 value_loss_coef: float, entropy_coef: float, lr: float = None, max_grad_norm: float = None,
                 use_clipped_value_loss=True, verbose=0):
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.num_grad_updates = num_grad_updates
        self.batch_size = batch_size
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)

        self.verbose = verbose

    def update(self, policy_buffer) -> dict:
        advantage = policy_buffer.returns[:-1] - policy_buffer.values[:-1]
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(self.num_grad_updates):

            data_generator = policy_buffer.get_batch_generator(self.batch_size, advantage)

            for sample in data_generator:
                states, actions, value_preds, returns, old_action_log_probs, adv_targets = \
                    sample['states'], sample['actions'], sample['values'], \
                    sample['returns'], sample['action_log_probs'], sample['adv_targets']

                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_action(states, actions)

                ratio = torch.exp(action_log_probs - old_action_log_probs)
                surr1 = ratio * adv_targets
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targets

                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds + \
                                         (values - value_preds).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - returns).pow(2)
                    value_losses_clipped = (
                            value_pred_clipped - returns).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.num_grad_updates * self.batch_size

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return {'value_loss': value_loss_epoch, 'action_loss': action_loss_epoch,
                'dist_entropy': dist_entropy_epoch}
