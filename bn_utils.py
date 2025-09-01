import logging
import torch.nn as nn


def _select_num_groups(channels):
    """Return a suitable ``GroupNorm`` group count for ``channels``.

    The largest power-of-two divisor not exceeding 32 is preferred. If no such
    divisor exists, the largest divisor \u2264 32 is used. A warning is logged
    when falling back to 1.
    """
    for ng in (32, 16, 8, 4, 2):
        if ng <= channels and channels % ng == 0:
            return ng
    for ng in range(min(32, channels), 1, -1):
        if channels % ng == 0:
            return ng
    logging.warning('Could not find a suitable GroupNorm divisor for %d channels; using 1 group', channels)
    return 1


def convert_batchnorm_modules(module, dp_mode='local'):
    """Replace ``BatchNorm`` layers with DP-friendly alternatives.

    Parameters
    ----------
    module : nn.Module
        Model to be converted in-place.
    dp_mode : str, optional
        Differential privacy mode. When ``'local'`` BatchNorm layers are
        replaced with ``GroupNorm``. For ``'server'`` or ``'off'`` BatchNorm
        layers are kept in evaluation mode.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.GroupNorm):
            convert_batchnorm_modules(child, dp_mode)
            continue

        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            if dp_mode != 'local':
                child.eval()
                convert_batchnorm_modules(child, dp_mode)
                continue

            num_groups = _select_num_groups(child.num_features)
            gn = nn.GroupNorm(num_groups, child.num_features, affine=child.affine, eps=child.eps)
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
            convert_batchnorm_modules(gn, dp_mode)
        else:
            convert_batchnorm_modules(child, dp_mode)
    return module

