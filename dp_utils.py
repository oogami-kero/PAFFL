import math
import torch


def compute_noisy_delta(global_params, local_params, clip_norm, noise_mult):
    """Compute clipped and noised updates for differential privacy."""
    delta = {}
    for k in global_params:
        if not torch.is_floating_point(global_params[k]):
            # integer buffers like num_batches_tracked are left unchanged
            continue
        delta[k] = local_params[k] - global_params[k]

    if not delta:
        return {}, {}

    vec = torch.cat([v.view(-1) for v in delta.values()])
    norm = torch.norm(vec)
    scale = min(1.0, clip_norm / (norm + 1e-12))

    for k in delta:
        delta[k] = delta[k] * scale

    delta_before_noise = {k: v.clone() for k, v in delta.items()}

    if noise_mult > 0:
        std = clip_norm * noise_mult
        for k in delta:
            delta[k] += torch.normal(0, std, size=delta[k].size(), device=delta[k].device)

    return delta, delta_before_noise


def compute_epsilon(num_steps, noise_mult, delta):
    """Approximate (epsilon, delta)-DP for Gaussian mechanism."""
    if noise_mult == 0:
        return float('inf')
    return math.sqrt(2 * num_steps * math.log(1 / delta)) / noise_mult
