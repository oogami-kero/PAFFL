import os
import sys
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dp_utils import compute_epsilon


def failing_accountant(*args, **kwargs):
    raise RuntimeError('fail')

with mock.patch('prv_accountant.Accountant', side_effect=failing_accountant):
    eps = compute_epsilon(10, 1.0, 1e-5, accountant='prv')
    print('fallback_eps', round(eps, 4))

def stub_accountant(*args, **kwargs):
    assert kwargs.get('mesh_size') == 0.01
    class Stub:
        def compute_epsilon(self, steps):
            return 0.5
    return Stub()

with mock.patch('prv_accountant.Accountant', side_effect=stub_accountant):
    eps = compute_epsilon(10, 1.0, 1e-5, accountant='prv', mesh_size=0.01)
    print('custom_eps', eps)
