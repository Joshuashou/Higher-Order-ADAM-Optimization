from Optimizer import Optimizer
from collections import OrderedDict, defaultdict, abc as container_abcs

class Higher_Moment_Adam(Optimizer):

    def __init__(self, params, num_moment = 2, lr=1e-2, betas=(0.9, 0.999, 0.999, 0.999, 0.999, 0.999), epsilon=1e-8,
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

            # print("Parameter groups")
            # print(group['params'])
            # sys.exit()


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
                    #EMA of cubed gradient values (Divergent Algo)
                    state['exp_avg_cub'] = grad.new().resize_as_(grad).zero_()
                    #EMA of Quartic gradient values
                    state['exp_avg_quart'] = grad.new().resize_as_(grad).zero_()
                    #EMA for 5 (Divergent)
                    state['exp_avg_five'] = grad.new().resize_as_(grad).zero_()
                    #EMA Sixth moment
                    state['exp_avg_six'] = grad.new().resize_as_(grad).zero_()

                exp_avgs = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_cub'], state['exp_avg_quart'], state['exp_avg_five'], state['exp_avg_six']
                
                beta_array = group['betas'] #Beta1, Beta2, Beta3, Beta4, Beta6

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient

                #We always keep exponential average of first Moment
                exp_avgs[0].mul_(beta_array[0]).add_(1-beta_array[0], grad)


                #Second Moving Average dependent on num_moment
                exp_avgs[self.num_moment - 1].mul_(beta_array[self.num_moment-1]).addcmul_(1 - beta_array[self.num_moment - 1],grad, grad ** (self.num_moment-1))

                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                # exp_avg_cub.mul_(beta3).add_(1 - beta3,grad ** 3)
                # exp_avg_quart.mul_(beta4).add_(1 - beta4, grad ** 4)
                # exp_avg_six.mul_(beta6).add_(1-beta6, grad ** 6)

                denom = (exp_avgs[self.num_moment - 1] ** (1./self.num_moment)).add_(group['epsilon'])
                # denom_sqrt = exp_avg_sq.sqrt().add_(group['epsilon'])
                # denom_cub = (exp_avg_cub ** 0.33).add_(group['epsilon'])
                # denom_quart = (exp_avg_quart ** 0.25).add_(group['epsilon'])
                # denom_six = (exp_avg_six ** (1./6)).add_(group['epsilon'])

                #Bias Correction for first estimate
                bias_correction1 = 1 - beta_array[0] ** state['step']

                #Bias Correction for num_moment
                bias_correction2 = 1 - beta_array[self.num_moment - 1] ** state['step']
                # bias_correction3 = 1 - beta3 ** state['step']
                # bias_correction4 = 1 - beta4 ** state['step']
                # bias_correction6 = 1 - beta6 ** state['step']


                step_size = group['lr'] * (bias_correction2 ** (1./self.num_moment)) / bias_correction1


                p.data.addcdiv_(-step_size, exp_avgs[0], denom)

        return loss

class Higher_Moment_Adam_Combination(Optimizer):

    def __init__(self, params,constants = [0.6,0.3,0.1], lr=1e-2, betas=(0.9, 0.999, 0.999, 0.999, 0.999, 0.999), epsilon=1e-8,
            weight_decay=0):
        defaults = dict(lr=lr, betas=betas, epsilon=epsilon,
                weight_decay=weight_decay)
        super(Higher_Moment_Adam_Combination, self).__init__(params, defaults)

        self.constants = constants


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
                    #EMA of cubed gradient values (Divergent Algo)
                    state['exp_avg_cub'] = grad.new().resize_as_(grad).zero_()
                    #EMA of Quartic gradient values
                    state['exp_avg_quart'] = grad.new().resize_as_(grad).zero_()
                    #EMA for 5 (Divergent)
                    state['exp_avg_five'] = grad.new().resize_as_(grad).zero_()
                    #EMA Sixth moment
                    state['exp_avg_six'] = grad.new().resize_as_(grad).zero_()

                exp_avgs = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_cub'], state['exp_avg_quart'], state['exp_avg_five'], state['exp_avg_six']
                
                beta_array = group['betas'] #Beta1, Beta2, Beta3, Beta4, Beta6

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient

                #We always keep exponential average of first Moment
                exp_avgs[0].mul_(beta_array[0]).add_(1-beta_array[0], grad)


                #Second Moving Average
                exp_avgs[1].mul_(beta_array[1]).addcmul_(1 - beta_array[1],grad, grad)

                #Fourth Moving Average
                exp_avgs[3].mul_(beta_array[3]).addcmul_(1 - beta_array[3],grad, grad ** 3)

                #Sixth Moving Average
                exp_avgs[5].mul_(beta_array[5]).addcmul_(1 - beta_array[5],grad, grad ** 5)


                #denom = (exp_avgs[self.num_moment - 1] ** (1./self.num_moment)).add_(group['epsilon'])
                denom_sqrt = exp_avgs[1].sqrt().add_(group['epsilon'])
                denom_quart = (exp_avgs[3] ** 0.25).add_(group['epsilon'])
                denom_six = (exp_avgs[5] ** (1./6)).add_(group['epsilon'])

                #Bias Correction for first estimate
                bias_correction1 = 1 - beta_array[0] ** state['step']

                #Bias Correction for num_moment
                bias_correction2 = 1 - beta_array[1] ** state['step']
                bias_correction4 = 1 - beta_array[2]  ** state['step']
                bias_correction6 = 1 - beta_array[3]  ** state['step']

                demon_sqrt = denom_sqrt * (bias_correction2 ** (1./2))
                denom_quart = denom_quart * (bias_correction4 ** (1./4))
                denom_six = denom_six * (bias_correction6 ** (1./6))

                denom = denom_sqrt * self.constants[0] + denom_quart * self.constants[1] + denom_six * self.constants[2]


                step_size = group['lr']# * (bias_correction2 ** (1./self.num_moment)) / bias_correction1


                p.data.addcdiv_(-step_size, exp_avgs[0], denom)

        return loss