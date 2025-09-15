import math
from opacus.optimizers.optimizer import (
    DPOptimizer,
    _generate_noise,
    _mark_as_processed,
    _check_processed_flag,
)


class LoggingDPOptimizer(DPOptimizer):
    """DPOptimizer that logs noise and gradient norms for each step.

    The optimizer behaves like :class:`opacus.optimizers.optimizer.DPOptimizer`
    but additionally records the L2 norm of the pre-noise gradient and the
    sampled noise for every optimisation step. These statistics are exposed via
    the ``grad_norm`` and ``noise_norm`` attributes and can be accumulated by
    the caller.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_norm = 0.0
        self.noise_norm = 0.0

    def add_noise(self):
        """Add DP noise while recording gradient and noise norms."""
        total_grad_sq = 0.0
        total_noise_sq = 0.0
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            grad = p.summed_grad
            total_grad_sq += grad.norm(2).item() ** 2
            noise = _generate_noise(
                std=self.noise_multiplier * self.max_grad_norm,
                reference=grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            total_noise_sq += noise.norm(2).item() ** 2
            p.grad = (grad + noise).view_as(p)
            _mark_as_processed(p.summed_grad)
        self.grad_norm = math.sqrt(total_grad_sq)
        self.noise_norm = math.sqrt(total_noise_sq)
