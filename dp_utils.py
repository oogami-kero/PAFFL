import math
import numpy as np
try:
    from opacus.grad_sample import GradSampleModule
except Exception:  # pragma: no cover - fallback when opacus is absent
    class GradSampleModule:  # type: ignore
        pass


def remove_dp_hooks(model):
    """Remove differential privacy hooks and cached gradients.

    This helper cleans up any Opacus hooks attached to ``model`` and deletes
    gradient sample attributes that may have been added during private
    training. If ``model`` is an instance of :class:`GradSampleModule` the base
    module is unwrapped before processing. Hooks on all submodules are removed
    and the underlying module is returned, making it safe to wrap again.

    Parameters
    ----------
    model : torch.nn.Module
        Model potentially wrapped in ``GradSampleModule``.

    Returns
    -------
    torch.nn.Module
        Unwrapped model with DP hooks removed.
    """
    if isinstance(model, GradSampleModule):
        model = model._module

    for submodule in model.modules():
        hooks = getattr(submodule, 'autograd_grad_sample_hooks', None)
        if hooks is not None:
            iterable = hooks.values() if isinstance(hooks, dict) else hooks
            for h in iterable:
                h.remove()
            delattr(submodule, 'autograd_grad_sample_hooks')
        for p in submodule.parameters(recurse=False):
            for attr in ('grad_sample', 'grad_sample_stack'):
                if hasattr(p, attr):
                    delattr(p, attr)

    return model


def compute_epsilon(num_steps, noise_mult, delta, accountant=None, sampling_rate=1.0):
    """Return an ``epsilon`` estimate for the Gaussian mechanism.

    Parameters
    ----------
    num_steps : int
        Total number of noisy updates that have been applied.
    noise_mult : float
        Noise multiplier used when generating the updates.
    delta : float
        Target ``delta`` parameter of differential privacy.
    accountant : str, optional
        If ``'rdp'`` an approximate R\u00E9nyi DP accountant is used for composition.
        If ``'prv'`` a privacy random variable accountant from ``prv_accountant``
        computes ε via a PLD representation. The accountant's
        ``compute_epsilon`` may return a scalar or a tuple ``(lower, estimate,
        upper)``; the estimate is used. Any other value falls back to a basic
        strong composition bound.
    sampling_rate : float, optional
        Probability that a given client participates in a round. Only used when
        ``accountant`` is ``'rdp'`` or ``'prv'``.

    Returns
    -------
    float
        Estimated privacy loss ε.
    """
    if noise_mult == 0:
        return float('inf')

    if accountant == 'rdp':
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = []
        for order in orders:
            rdp.append(num_steps * (sampling_rate ** 2) * order / (2 * noise_mult ** 2))
        eps = min(r - math.log(delta) / (o - 1) for r, o in zip(rdp, orders))
        return eps

    if accountant == 'prv':
        from prv_accountant import Accountant

        accountant = Accountant(
            noise_multiplier=noise_mult,
            sampling_probability=sampling_rate,
            delta=delta,
            max_compositions=num_steps,
            eps_error=0.1,
        )
        eps = accountant.compute_epsilon(num_steps)
        return eps if isinstance(eps, float) else eps[1]

    return math.sqrt(2 * num_steps * math.log(1 / delta)) / noise_mult


def scale_noise_to_clip(noise_mult, old_clip, new_clip):
    """Return a noise multiplier keeping the noise-to-clip ratio fixed.

    Parameters
    ----------
    noise_mult : float
        Current noise multiplier.
    old_clip : float
        Previous clipping bound.
    new_clip : float
        Updated clipping bound.

    Returns
    -------
    float
        Noise multiplier rescaled so that ``noise_mult / old_clip`` equals
        ``new_noise / new_clip``.
    """
    if old_clip == 0:
        raise ValueError('old_clip must be non-zero')
    return (noise_mult / old_clip) * new_clip


def find_noise_multiplier(
    num_steps,
    target_eps,
    delta,
    accountant=None,
    sampling_rate=1.0,
    sigma_min=0.5,
    sigma_max=10.0,
    tol=0.05,
    max_iter=50,
):
    """Binary search for a noise multiplier that meets a target ``epsilon``.

    Parameters
    ----------
    num_steps : int
        Total number of noisy updates that will be applied.
    target_eps : float
        Desired privacy guarantee.
    delta : float
        Target ``delta`` parameter of differential privacy.
    accountant : str, optional
        Accounting method passed through to :func:`compute_epsilon`.
    sampling_rate : float, optional
        Client participation probability when ``accountant`` is ``'rdp'`` or ``'prv'``.
    sigma_min, sigma_max : float, optional
        Search range for the noise multiplier.
    tol : float, optional
        Tolerance for the returned ``epsilon``.
    max_iter : int, optional
        Maximum number of search iterations.

    Returns
    -------
    float
        Noise multiplier that yields ``epsilon`` within ``tol`` of ``target_eps``.
    """

    def eps_for(sigma):
        return compute_epsilon(num_steps, sigma, delta, accountant, sampling_rate)

    eps_low = eps_for(sigma_min)
    eps_high = eps_for(sigma_max)
    if eps_low < target_eps or eps_high > target_eps:
        raise ValueError('Target epsilon is not bracketed by search range')

    lo, hi = sigma_min, sigma_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        eps = eps_for(mid)
        if eps > target_eps:
            lo = mid
        else:
            hi = mid
        if abs(eps - target_eps) <= tol:
            break
    return hi


def weighted_median(values, weights=None):
    """Return the weighted median of ``values``.

    Parameters
    ----------
    values : list[float]
        Data values.
    weights : list[float], optional
        Corresponding non-negative weights. If ``None`` all weights are equal.

    Returns
    -------
    float
        Weighted median.
    """
    if not values:
        raise ValueError('values must be non-empty')
    if weights is None:
        weights = [1.0] * len(values)
    sorted_pairs = sorted(zip(values, weights), key=lambda x: x[0])
    values, weights = zip(*sorted_pairs)
    cum_weights = np.cumsum(weights)
    threshold = 0.5 * cum_weights[-1]
    idx = int(np.searchsorted(cum_weights, threshold, side='left'))
    return float(values[idx])


def initial_clip(p90s, weights=None, min_clip=1.2, max_clip=5.0):
    """Compute the initial clipping bound from client percentiles."""
    clip = weighted_median(p90s, weights)
    return float(max(min_clip, min(clip, max_clip)))


class AdaptiveClipper:
    """Adaptive clipping controller with percentile blending.

    The controller adjusts the clipping bound to keep the fraction of clipped
    client updates close to a target level while incorporating percentile
    information from gradient norms. Noise is assumed to remain fixed.
    """

    def __init__(
        self,
        clip,
        target=0.2,
        beta=0.95,
        kp=0.05,
        gamma=0.1,
        alpha=0.5,
        min_clip=1.2,
        max_clip=5.0,
    ):
        self.clip = clip
        self.target = target
        self.beta = beta
        self.kp = kp
        self.gamma = gamma
        self.alpha = alpha
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.clip_frac_ema = target
        self._deadband = 0.05
        self._steady_rounds = 0
        self._high_ctr = 0
        self._low_ctr = 0

    def update(self, clipped_fraction, q90):
        """Update the clipping bound.

        Parameters
        ----------
        clipped_fraction : float
            Fraction of client updates that were clipped this round.
        q90 : float
            90th percentile of pre-clip gradient norms across clients.

        Returns
        -------
        float
            Updated clipping bound.
        """
        self.clip_frac_ema = self.beta * self.clip_frac_ema + (1 - self.beta) * clipped_fraction
        if abs(self.clip_frac_ema - self.target) <= self._deadband:
            self._steady_rounds += 1
        else:
            self._steady_rounds = 0
        if self._steady_rounds >= 3:
            return self.clip

        log_c = math.log(self.clip)
        log_c_prime = log_c + self.kp * (self.clip_frac_ema - self.target)
        log_c_tilde = (1 - self.gamma) * log_c + self.gamma * math.log(max(q90, 1e-12))
        log_c_new = (1 - self.alpha) * log_c_prime + self.alpha * log_c_tilde
        c_new = float(math.exp(log_c_new))
        c_new = max(min(c_new, self.clip * 1.1), self.clip * 0.9)
        c_new = max(self.min_clip, min(c_new, self.max_clip))
        self.clip = c_new

        if clipped_fraction > 0.5:
            self._high_ctr += 1
            self._low_ctr = 0
        elif clipped_fraction < 0.05:
            self._low_ctr += 1
            self._high_ctr = 0
        else:
            self._high_ctr = 0
            self._low_ctr = 0

        if self._high_ctr >= 5:
            self.clip = min(self.clip * 1.12, self.max_clip)
            self._high_ctr = 0
        if self._low_ctr >= 5:
            self.clip = max(self.clip * 0.88, self.min_clip)
            self._low_ctr = 0

        return self.clip
