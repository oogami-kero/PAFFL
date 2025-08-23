import math
from opacus.grad_sample import GradSampleModule


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
