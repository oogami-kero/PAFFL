import torch.nn as nn


def convert_batchnorm_modules(module):
    """Replace all ``nn.BatchNorm2d`` layers with ``nn.GroupNorm``.

    Parameters
    ----------
    module : nn.Module
        Model to be converted in-place.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(1, child.num_features, affine=child.affine, eps=child.eps)
            setattr(module, name, gn)
        else:
            convert_batchnorm_modules(child)
    return module

