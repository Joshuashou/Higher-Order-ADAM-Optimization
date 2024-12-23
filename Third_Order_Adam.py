from Optimizer import Optimizer
from collections import OrderedDict, defaultdict, abc as container_abcs


#Demonstrates intractability of third order moment, the Skew causes the gradient to be unstable. 
import math
import torch


class ThirdOrderAdam(Optimizer):
    r"""
    Example of a "third-order" Adam-like optimizer that accumulates:
      1. First moment (EMA of gradients)        -> exp_avg
      2. Second moment (EMA of squared grads)   -> exp_avg_sq
      3. Third moment (EMA of cubed grads)      -> exp_avg_cu

    The update rule is a toy example. Real usage might tweak how these moments
    are combined or how the step_size is adjusted.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999, 0.9999),  # beta1, beta2, beta3
        eps=1e-8,
        weight_decay=0
    ):
        """
        :param params: Iterable of parameters to optimize (typically model.parameters()).
        :param lr: Learning rate.
        :param betas: Tuple of (beta1, beta2, beta3).
        :param eps: Term added to denominator to improve numerical stability.
        :param weight_decay: Weight decay (L2 penalty).
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if any(b < 0.0 or b >= 1.0 for b in betas):
            raise ValueError(f"Invalid beta parameter(s): {betas}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        :param closure: A closure that reevaluates the model and returns the loss.
        :return: The loss (if closure is provided), otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("ThirdOrderAdam does not support sparse gradients.")

                # Weight decay
                if weight_decay != 0.0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # 1st moment of gradients
                    state["exp_avg"] = grad.new().resize_as_(grad).zero_()
                    # 2nd moment of gradients
                    state["exp_avg_sq"] = grad.new().resize_as_(grad).zero_()
                    # 3rd moment of gradients
                    state["exp_avg_cu"] = grad.new().resize_as_(grad).zero_()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_cu = state["exp_avg_cu"]

                state["step"] += 1

                # Decay the moving averages
                # 1st moment
                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
                # 2nd moment
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                # 3rd moment
                # Example: EMA of (grad^3). Alternatively, grad * grad^2, etc.
                # The "addcmul_" call multiplies the last two arguments (grad, grad**2)
                # then multiplies that by (1-beta3), then adds to exp_avg_cu * beta3.
                grad_sq = grad * grad
                exp_avg_cu.mul_(beta3).addcmul_(grad, grad_sq, value=(1 - beta3))

                # Bias corrections
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                bias_correction3 = 1 - beta3 ** state["step"]  # if you want it

                # For standard Adam, denom = sqrt(exp_avg_sq) + eps

                # Example: Weighted combo of sqrt(2nd moment) and cbrt(3rd moment)
                second = (exp_avg_sq / bias_correction2).sqrt()
                third  = (exp_avg_cu / max(bias_correction3, 1e-16)).abs().pow(1.0 / 3.0)

                alpha_for_third = 0.5  # a weighting factor
                denom = second.add(third, alpha=alpha_for_third).add_(eps)
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
