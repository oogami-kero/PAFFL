import math
import logging
import numpy as np
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

def weighted_median(values, weights):
    '''Return the weighted median of ``values`` with corresponding ``weights``.'''
    if not values:
        return 0.0
    values = np.asarray(values)
    weights = np.asarray(weights)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cutoff = 0.5 * cdf[-1]
    return float(values[cdf >= cutoff][0])


def stabilize_adaptive_clip(
    args,
    epsilon,
    num_clients,
    round_p90,
    round_p50=None,
    frac_now=None,
    logger=logging,
):
    '''Update the DP-SGD clipping bound using a stabilized controller.

    Parameters
    ----------
    args : argparse.Namespace
        Namespace holding DP-related state such as dp_clip and
        last_clip_fraction. This function updates dp_clip in-place.
    epsilon : float
        Current cumulative privacy loss.
    num_clients : int
        Number of participating clients in the current round.
    round_p90 : float
        90th percentile of gradient norms from the current round.
    round_p50 : float, optional
        Median of gradient norms from the current round.
    frac_now : float, optional
        Fraction of client norms exceeding the clipping bound.
    logger : logging.Logger, optional
        Logger used for reporting statistics.

    Returns
    -------
    float
        Updated clipping bound.
    '''
    if round_p50 is not None:
        args.round_p50 = round_p50
    if frac_now is not None:
        args.frac_now = frac_now
    p_t = max(0.02, min(0.98, getattr(args, 'last_clip_fraction', 0.0)))
    if not hasattr(args, 'clip_frac_ema'):
        args.clip_frac_ema = p_t
    else:
        args.clip_frac_ema = 0.95 * args.clip_frac_ema + 0.05 * p_t
    target = getattr(args, 'dp_target_clip_fraction', 0.1)
    if abs(args.clip_frac_ema - target) <= 0.05:
        args.deadband_ctr = getattr(args, 'deadband_ctr', 0) + 1
    else:
        args.deadband_ctr = 0
    if getattr(args, 'deadband_ctr', 0) >= 3:
        new_clip = args.dp_clip
    else:
        log_c = math.log(args.dp_clip)
        log_ctrl = log_c + 0.05 * (args.clip_frac_ema - target)
        perc = max(round_p90, 1e-12)
        log_perc = 0.9 * log_c + 0.1 * math.log(perc)
        log_blend = 0.5 * log_ctrl + 0.5 * log_perc
        cand = math.exp(log_blend)
        step = getattr(args, 'dp_clip_step_cap', 0.10)
        low_step = args.dp_clip * (1.0 - step)
        high_step = args.dp_clip * (1.0 + step)
        cand = max(low_step, min(high_step, cand))
        dp_min = getattr(args, 'dp_clip_min', 1.2)
        dp_max = getattr(args, 'dp_clip_max', 5.0)
        new_clip = max(dp_min, min(dp_max, cand))
    args.dp_clip = new_clip
    args.log_dp_clip = math.log(args.dp_clip)
    noise_std = args.dp_noise * args.dp_noise_scale / num_clients
    if getattr(args, 'last_clip_fraction', 0.0) > 0.95:
        args.saturation_ctr = getattr(args, 'saturation_ctr', 0) + 1
    else:
        args.saturation_ctr = 0
    if getattr(args, 'saturation_ctr', 0) >= 3 and args.dp_clip >= 0.98 * getattr(args, 'dp_clip_max', args.dp_clip):
        msg = (
            'Clipped ~100%, controller saturated at dp_clip_max. '
            'Raise --dp_clip_max (try ×1.5) or reduce local drift (epochs/μ).'
        )
        print(f'WARNING: {msg}')
        logger.warning(msg)
    print(
        f"clip EMA: {args.clip_frac_ema:.4f}, hist q90: {getattr(args, 'q90', 0.0):.4f}, round p90: {round_p90:.4f}, DP clip: {args.dp_clip:.4f}, noise std: {noise_std:.4f}, eps: {epsilon or 0.0:.4f}"
    )
    logger.info(
        'DP clip %.4f, clip EMA %.4f, hist q90 %.4f, round p90 %.4f, noise std %.4f, eps %.4f',
        args.dp_clip,
        args.clip_frac_ema,
        getattr(args, 'q90', 0.0),
        round_p90,
        noise_std,
        epsilon or 0.0,
    )
    return args.dp_clip
