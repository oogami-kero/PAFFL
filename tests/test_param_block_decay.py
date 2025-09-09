import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dp_utils import get_param_block

BLOCK_DEPTH = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3, 'fc': 3, 'head': 3}


def test_prefixed_module_decay_applies():
    """Parameters with prefixes should decay based on internal block."""
    name = 'encoder.layer3.0.conv.weight'
    decay = 0.5
    base_target = 1.0
    block = get_param_block(name)
    depth = BLOCK_DEPTH.get(block, 0)
    target = base_target * (decay ** depth)
    assert target == base_target * (decay ** 2)
