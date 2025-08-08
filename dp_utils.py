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
        If set to ``"rdp"`` an approximate R\u00E9nyi DP accountant is used for
        composition. Otherwise a basic strong composition bound is used.
    sampling_rate : float, optional
        Probability that a given client participates in a round. This is only
        used when ``accountant='rdp'``.
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

    return math.sqrt(2 * num_steps * math.log(1 / delta)) / noise_mult
