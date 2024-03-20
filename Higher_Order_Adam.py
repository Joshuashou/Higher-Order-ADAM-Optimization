from Optimizer import Optimizer
from collections import OrderedDict, defaultdict, abc as container_abcs

class Higher_Moment_Adam(Optimizer):

    def __init__(self, params, num_moment = 4, lr=1e-2, betas=(0.9, 0.999), epsilon=1e-8,
            weight_decay=0):
        defaults = dict(lr=lr, betas=betas, epsilon=epsilon,
                weight_decay=weight_decay)
        super(Higher_Moment_Adam, self).__init__(params, defaults)
        self.num_moment = num_moment

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
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient value
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                    state['exp_avg_quart'] = grad.new().resize_as_(grad).zero_()


                exp_avg, exp_avg_4 = state['exp_avg'], state['exp_avg_quart']

                beta_1, beta_2 = group['betas'] #Beta1, Beta2, Beta4

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient

                #We always keep exponential average of first Moment
                exp_avg.mul_(beta_1).add_(1-beta_1, grad)
                exp_avg_4.mul_(beta_2).addcmul_(1-beta_2, grad, grad ** 3)

                bias_correction1 = 1 - beta_1 ** state['step']
                bias_correction2 = 1 - beta_2 ** state['step']

                # Modify here: Adjust step size with the fourth root
                step_size = group['lr'] * (bias_correction2 ** 0.25) / bias_correction1


                # Modify here: Update parameters with the fourth root
                denom = exp_avg_4.sqrt().sqrt().add_(group['epsilon'])
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

