import numpy as np
import torch

from slbo.models import Actor, VCritic
from slbo.models.utils import get_flat_params, set_flat_params
from slbo.misc import logger


# noinspection DuplicatedCode
class TRPO:
    def __init__(self, actor: Actor, critic: VCritic, critic_lr=3e-4, max_kld=1e-2, l2_reg_coef=1e-3, damping=0.1,
                 entropy_coef=0., line_search_accepted_ratio=0.1, verbose=0):

        self.actor = actor
        self.critic = critic

        self.max_kld = max_kld
        self.l2_reg_coef = l2_reg_coef
        self.damping = damping
        self.linesearch_accepted_ratio = line_search_accepted_ratio
        self.entropy_coef = entropy_coef

        self.critic_lr = critic_lr
        self.num_critic_updates = 5
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self.critic_lr)
        self.verbose = verbose

    @staticmethod
    def get_conjugate_gradient(Avp, b, nsteps, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = Avp(p)
            alpha = rdotr / torch.dot(p, _Avp)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def linesearch(self, f, init_params, fullstep, expected_improve_rate, max_backtracks=10):
        with torch.no_grad():
            fval = f()
            for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
                new_params = init_params + stepfrac * fullstep
                set_flat_params(self.actor, new_params)
                newfval = f()
                actual_improve = fval - newfval
                expected_improve = expected_improve_rate * stepfrac
                ratio = actual_improve / expected_improve
                if self.verbose > 0:
                    logger.log("a/e/r ", actual_improve.item(), expected_improve.item(), ratio.item())
                if ratio.item() > self.linesearch_accepted_ratio and actual_improve.item() > 0:
                    return True, new_params
            return False, init_params

    def update_critc(self, states, targets):
        values = self.critic(states)
        value_loss_ = (values - targets).pow(2).mean()

        loss = value_loss_
        for param in self.critic.parameters():
            loss += param.pow(2).sum() * self.l2_reg_coef
        self.critic_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2)
        self.critic_optim.step()
        return loss

    def update(self, policy_buffer) -> dict:
        advantages = policy_buffer.returns[:-1] - policy_buffer.values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        samples = next(policy_buffer.get_batch_generator(advantages=advantages))

        states, actions, adv_targets = samples['states'], samples['actions'], samples['adv_targets']

        fixed_log_prob = self.actor.evaluate_action(states, actions)[0].detach()

        def get_action_loss():
            log_prob, entropy = self.actor.evaluate_action(states, actions)
            action_loss_ = - adv_targets * torch.exp(log_prob - fixed_log_prob) - self.entropy_coef * entropy
            return action_loss_.mean()

        def get_kl():
            *_, action_means, action_logstds, action_stds = self.actor.act(states)

            fixed_action_means = action_means.detach()
            fixed_action_logstds = action_logstds.detach()
            fixed_action_stds = action_stds.detach()
            kl = action_logstds - fixed_action_logstds + \
                 (fixed_action_stds.pow(2) + (fixed_action_means - action_means).pow(2)) / \
                 (2.0 * action_stds.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        action_loss = get_action_loss()

        action_loss_grad = torch.autograd.grad(action_loss, self.actor.parameters())
        flat_action_loss_grad = torch.cat([grad.view(-1) for grad in action_loss_grad]).data

        def Fvp(v):
            kl = get_kl()
            kl = kl.mean()

            kld_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
            flat_kld_grad = torch.cat([grad.view(-1) for grad in kld_grad])

            kl_v = (flat_kld_grad * v).sum()
            kld_grad_grad = torch.autograd.grad(kl_v, self.actor.parameters())
            flat_kld_grad_grad = torch.cat([grad.contiguous().view(-1) for grad in kld_grad_grad]).data

            return flat_kld_grad_grad + v * self.damping

        stepdir = self.get_conjugate_gradient(Fvp, -flat_action_loss_grad, 10)

        shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0)

        lm = torch.sqrt(shs / self.max_kld)
        fullstep = stepdir / lm

        neggdotstepdir = (-flat_action_loss_grad * stepdir).sum(0, keepdim=True)
        if self.verbose > 0:
            logger.info(("lagrange multiplier:", lm, "grad_norm:", flat_action_loss_grad.norm()))

        prev_params = get_flat_params(self.actor)
        success, new_params = self.linesearch(get_action_loss, prev_params, fullstep, neggdotstepdir / lm)
        set_flat_params(self.actor, new_params)

        action_loss_epoch = action_loss.item()

        value_loss_epoch = 0.
        gen = policy_buffer.get_batch_generator(batch_size=64)

        num_updates = 0
        for _ in range(self.num_critic_updates):
            for samples in gen:
                num_updates += 1
                states, returns = samples['states'], samples['returns']
                value_loss = self.update_critc(states, returns)
                value_loss_epoch += value_loss.item()
        value_loss_epoch /= num_updates

        return {'action_loss': action_loss_epoch, 'value_loss': value_loss_epoch}
