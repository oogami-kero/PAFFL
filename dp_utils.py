import math
import torch


def compute_noisy_delta(global_params, local_params, clip_norm, noise_mult):
    """Compute clipped and noised updates for differential privacy.

    Parameters whose names begin with ``"transform_layer."`` are skipped so that
    personalized components are never aggregated or shared.
    """
    delta = {}
    for k in global_params:
        if k.startswith("transform_layer."):
            continue
        if not torch.is_floating_point(global_params[k]):
            # integer buffers like num_batches_tracked are left unchanged
            continue
        delta[k] = local_params[k] - global_params[k]

    if not delta:
        return {}, {}

    vec = torch.cat([v.view(-1) for v in delta.values()])
    norm = torch.norm(vec)

    print(f"DEBUG: Unclipped Delta Norm = {norm.item()}")

    scale = min(1.0, clip_norm / (norm + 1e-12))

    for k in delta:
        delta[k] = delta[k] * scale

    delta_before_noise = {k: v.clone() for k, v in delta.items()}

    if noise_mult > 0:
        std = clip_norm * noise_mult
        for k in delta:
            delta[k] += torch.normal(0, std, size=delta[k].size(), device=delta[k].device)

    return delta, delta_before_noise


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
