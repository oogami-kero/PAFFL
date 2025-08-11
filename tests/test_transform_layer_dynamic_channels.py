import os
import sys

import torch
import pytest
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Stub opacus to avoid heavy dependency during import
opacus = types.ModuleType('opacus')
validators = types.ModuleType('opacus.validators')


class _DummyValidator:
    @staticmethod
    def validate(module):
        return module


validators.ModuleValidator = _DummyValidator
opacus.validators = validators
sys.modules['opacus'] = opacus
sys.modules['opacus.validators'] = validators
from model import TransformLayer


def test_transform_layer_initializes_on_first_forward():
    layer = TransformLayer()
    x = torch.randn(2, 4, 3, 3)
    out = layer(x)
    assert out.shape == x.shape
    assert layer.alpha.shape[1] == 4
    assert layer.beta.shape[1] == 4
    x_same = torch.randn(2, 4, 3, 3)
    layer(x_same)
    x_diff = torch.randn(2, 5, 3, 3)
    with pytest.raises(ValueError):
        layer(x_diff)

