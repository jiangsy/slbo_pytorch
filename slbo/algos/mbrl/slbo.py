from operator import itemgetter

import torch

from slbo.models.dynamics import Dynamics
from slbo.models.normalizer import Normalizer


class SLBO:
    def __init__(self, dynamics: Dynamics, normalizers: Normalizer, batch_size: int, num_updates: int,
                 num_rollout_steps=4, l2_reg_coef=0.01, max_grad_norm=10, lr=3e-4):
        self.dynamics = dynamics
        self.normalizers = normalizers

        self.num_updates = num_updates
        self.num_rollout_steps = num_rollout_steps
        self.batch_size = batch_size
        self.l2_reg_coef = l2_reg_coef
        self.max_grad_norm = max_grad_norm

        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr)

    def update(self, model_buffer) -> dict:

        gen = model_buffer.get_sequential_batch_generator(self.batch_size, self.num_rollout_steps)

        model_loss_epoch = 0.
        l2_loss_epoch = 0.
        for _ in range(self.num_updates):
            try:
                state_sequences, action_sequences, next_state_sequences, mask_sequences = \
                    itemgetter(*['states', 'actions', 'next_states', 'masks'])(next(gen))
            except StopIteration:
                gen = model_buffer.get_sequential_batch_generator(self.batch_size, self.num_rollout_steps)
                state_sequences, action_sequences, next_state_sequences, mask_sequences = \
                    itemgetter(*['states', 'actions', 'next_states', 'masks'])(next(gen))

            cur_states = state_sequences[:, 0]
            model_loss = 0.

            for i in range(self.num_rollout_steps):
                next_states = self.dynamics(cur_states, action_sequences[:, i])
                diffs = next_states - cur_states - next_state_sequences[:, i] + state_sequences[:, i]
                weighted_diffs = diffs / torch.clamp(self.normalizers.diff_normalizer.std, min=1e-6)
                model_loss += weighted_diffs.pow(2).sum(-1).mean()

                if i < self.num_rollout_steps - 1:
                    cur_states = state_sequences[:, i + 1] + \
                                 mask_sequences[:, i] * (next_states - state_sequences[:, i + 1])

            params = self.dynamics.parameters()
            l2_loss = self.l2_reg_coef * torch.stack([torch.norm(t, p=2) for t in params]).sum()

            model_loss_epoch += model_loss.item()
            l2_loss_epoch += l2_loss.item()

            self.dynamics_optimizer.zero_grad()
            (model_loss / self.num_rollout_steps + l2_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.max_grad_norm)
            self.dynamics_optimizer.step()

        model_loss_epoch /= self.num_updates
        return {'model_loss': model_loss_epoch, 'l2_loss': l2_loss_epoch}
