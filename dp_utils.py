import math

try:
    from opacus.grad_sample import GradSampleModule
except Exception:  # pragma: no cover - optional dependency
    class GradSampleModule:  # type: ignore
        """Fallback stub when Opacus is unavailable."""
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


def compute_epsilon(num_steps, noise_mult, delta, accountant=None, sampling_rate=1.0, mesh_size=None):
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
        computes ε via a PLD representation. If the high-level accountant
        construction fails due to tiny noise multipliers, a discretised
        low-level API is used, falling back to an approximate R\u00E9nyi DP bound
        if PRV evaluation is still not feasible. The accountant's
        ``compute_epsilon`` may return a scalar or a tuple ``(lower, estimate,
        upper)``; the estimate is used. Any other value falls back to a basic
        strong composition bound.
    sampling_rate : float, optional
        Probability that a given client participates in a round. Only used when
        ``accountant`` is ``'rdp'`` or ``'prv'``.
    mesh_size : float, optional
        Resolution of the numerical integration grid used by the PRV accountant.
        If ``None`` a heuristic of ``noise_mult / 10`` is applied.

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
        mesh = noise_mult / 10 if mesh_size is None else mesh_size
        try:
            from prv_accountant import Accountant

            acc = Accountant(
                noise_multiplier=noise_mult,
                sampling_probability=sampling_rate,
                delta=delta,
                max_compositions=num_steps,
                eps_error=0.1,
                mesh_size=mesh,
            )
            eps = acc.compute_epsilon(num_steps)
        except (AssertionError, RuntimeError):
            try:
                from prv_accountant import accountant as acc_mod, prv, discretisers

                tprv = prv.GaussianMechanism(
                    noise_multiplier=noise_mult,
                    sampling_probability=sampling_rate,
                )
                domain = acc_mod.Domain(-50 * noise_mult, 50 * noise_mult)
                disc = discretisers.ExplicitDomain(domain=domain, mesh_size=mesh)
                acc = acc_mod.PRVAccountant(
                    prvs=[tprv],
                    discretiser=disc,
                    max_compositions=num_steps,
                    eps_error=0.1,
                    delta=delta,
                )
                eps = acc.compute_epsilon(num_steps)
            except Exception:
                return compute_epsilon(num_steps, noise_mult, delta, 'rdp', sampling_rate)
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
