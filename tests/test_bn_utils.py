import os
import sys
import torch.nn as nn
from opacus.validators import ModuleValidator

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from bn_utils import convert_batchnorm_modules


class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 48, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(48)

    def forward(self, x):
        return self.bn(self.conv(x))


m = Dummy()
convert_batchnorm_modules(m, dp_mode='local')
convert_batchnorm_modules(m, dp_mode='local')
ModuleValidator.validate(m, strict=True)
print('num_groups', m.bn.num_groups)
