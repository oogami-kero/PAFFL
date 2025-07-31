import math

class PrivacyAccountant:
    """Simple RDP accountant for DP-SGD.

    The implementation uses an analytical upper bound from the moments
    accountant presented in *Deep Learning with Differential Privacy*.
    It provides a rough estimate of the current :math:`\varepsilon` given the
    noise multiplier, clipping norm and number of optimisation steps.
    """

    def __init__(self, noise_multiplier, clipping_norm, batch_size, dataset_size, target_delta=1e-5):
        """Create a new accountant instance."""
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.target_delta = target_delta
        self.steps = 0

    def step(self, sample_size=None):
        """Record one optimisation step."""
        self.steps += 1
        if sample_size is not None:
            self.batch_size = sample_size

    def get_epsilon(self):
        """Return the current :math:`\varepsilon` value."""
        if self.noise_multiplier == 0:
            return float('inf')
        q = self.batch_size / float(self.dataset_size)
        if q > 1:
            q = 1.0
        eps = q * math.sqrt(2 * self.steps * math.log(1 / self.target_delta)) / self.noise_multiplier
        eps += (self.steps * q ** 2) / (self.noise_multiplier ** 2)
        return eps
