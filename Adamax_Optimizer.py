#Use similar methods on the L-Infinity Norm, of AdaMax 
from Optimizer import Optimizer
import torch

class Adamax(Optimizer):

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adamax, self).__init__(params, defaults)

    def step(self, forward_closure=None):
        loss = None
        if forward_closure is not None:
            loss = self._forward_backward(forward_closure)

        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad
                state = self.state[id(p)]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    state['exp_inf'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']
                eps = group['eps']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Update biased first moment estimate.
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # Update the exponentially weighted infinity norm.
                norm_buf = torch.cat([
                    exp_inf.mul_(beta2).unsqueeze(0),
                    grad.abs().add_(eps).unsqueeze_(0)
                ], 0)
                state['exp_inf'] = exp_inf = (torch.max(norm_buf, 0)[0]).squeeze_(0)

                bias_correction = 1 - beta1 ** state['step']
                clr = group['lr'] / bias_correction

                p.data.addcdiv_(-clr, exp_avg, exp_inf)

        return loss
